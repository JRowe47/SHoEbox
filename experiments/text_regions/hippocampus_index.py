"""Bounded binary directory + sparse postings; no full record/region scans.

Storage is a lazy dictionary of immutable one-record pages. Query copies no bank;
commits shallow-copy only touched metadata. Tensor pages are shared by snapshots.
This Python reference is not a distributed backend or a throughput claim.
"""
from dataclasses import dataclass, field
import heapq
import math


@dataclass
class WorkBudget:
    config: object
    shards: set = field(default_factory=set)
    postings: dict = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)
    links: set = field(default_factory=set)
    directory_nodes: int = 0
    stale: int = 0
    exploration: int = 0
    routing_miss: int = 0
    refinements: int = 0

    def visit_shard(self, shard):
        if shard in self.shards:
            return True
        if len(self.shards) >= self.config.shards_per_request:
            return False
        self.shards.add(shard)
        return True

    def posting(self, shard):
        n = self.postings.get(shard, 0)
        if n >= self.config.posting_reads:
            return False
        self.postings[shard] = n+1
        return True

    def candidate(self, handle):
        if not self.visit_shard(handle.shard):
            return False
        bucket = self.candidates.setdefault(handle.shard, set())
        if handle in bucket:
            return False
        if len(bucket) >= self.config.candidates_per_shard:
            return False
        bucket.add(handle)
        return True

    def read(self, handle):
        if handle in self.links or len(self.links) >= self.config.readbacks:
            return False
        self.links.add(handle)
        return True

    def metrics(self):
        return dict(shards=len(self.shards), posting_reads=sum(self.postings.values()),
                    candidates=sum(map(len,self.candidates.values())), readbacks=len(self.links),
                    regions=len({h.region for h in self.links}), directory_nodes=self.directory_nodes,
                    stale=self.stale, exploration=self.exploration, routing_miss=self.routing_miss,
                    refinements=self.refinements)


class SparseDirectory:
    def __init__(self, config):
        self.config = config

    def route(self, key, budget, sequence=0, explore=True):
        depth = int(math.log2(self.config.shards))
        values = key.detach().tolist()
        queue, leaves = [(0., 0, 0)], []
        while queue and len(leaves) < self.config.shards_per_request:
            cost, negative_level, prefix = heapq.heappop(queue)
            level = -negative_level
            budget.directory_nodes += 1
            if level == depth:
                leaves.append(prefix)
                continue
            for bit in (0,1):
                penalty = max(0., (1-2*bit)*values[level])
                # Prefer deeper equal-cost paths. A zero/ambiguous address must
                # not turn best-first search into a breadth-first shard scan.
                heapq.heappush(queue,(cost+penalty,-level-1,2*prefix+bit))
        # Replace the last route, never add an unbudgeted retry. Counter-based
        # exploration consumes no global RNG and is exactly checkpointable.
        if explore and sequence % self.config.exploration_interval == 0 and len(leaves) < self.config.shards:
            other = (sequence//self.config.exploration_interval) % self.config.shards
            if other not in leaves:
                leaves[-1] = other
                budget.exploration += 1
        return [s for s in leaves if budget.visit_shard(s)]

    def lookup(self, row, key, cells, watermark, budget, sequence=0):
        result, seen = [], set()
        for shard in self.route(key,budget,sequence):
            # Leave room for neighbor expansion inside the original candidate cap.
            initial = max(1,self.config.candidates_per_shard-self.config.neighbors//2)
            count = 0
            postings = [row.postings.get((shard,int(cell)),()) for cell in cells]
            if self.config.bootstrap:
                postings.append(row.recent.get(shard,()))
            for posting in postings:
                for handle in reversed(posting):
                    if not budget.posting(shard):
                        break
                    if handle in seen:
                        continue
                    seen.add(handle)
                    record = row.binding(handle,watermark)
                    if record is None:
                        budget.stale += 1
                        continue
                    if budget.candidate(handle):
                        result.append(record)
                        count += 1
                    if count >= initial:
                        break
                if count >= initial or budget.postings.get(shard,0) >= self.config.posting_reads:
                    break
        if not result:
            budget.routing_miss += 1
        return result

    def expand(self, row, records, watermark, budget):
        result = list(records)
        for record in records:
            for handle in record.neighbors:
                if budget.candidate(handle):
                    neighbor = row.binding(handle,watermark)
                    if neighbor is None:
                        budget.stale += 1
                    else:
                        result.append(neighbor)
        return result

    def install(self, row, record, old=None):
        """Touched postings only; expired identities can never alias recycled slots."""
        postings, recent = dict(row.postings), dict(row.recent)
        if old is not None:
            for cell in old.code.detach().nonzero().flatten().tolist():
                key = (old.handle.shard,cell)
                value = tuple(h for h in postings.get(key,()) if h != old.handle)
                if value:
                    postings[key] = value
                else:
                    postings.pop(key,None)
        for cell in record.code.detach().nonzero().flatten().tolist():
            key = (record.handle.shard,cell)
            postings[key] = (*postings.get(key,()),record.handle)[-self.config.posting_reads:]
        key = record.handle.shard
        recent[key] = tuple(h for h in recent.get(key,()) if old is None or h != old.handle)
        recent[key] = (*recent[key],record.handle)[-self.config.posting_reads:]
        return postings,recent
