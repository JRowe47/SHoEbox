"""A9 training-only nuisance substitution; inference never imports this helper.

Exact rule: reports/exp0305a8/A9_REGISTRATION.md. No existing RNG is consumed.
"""
import hashlib
from dataclasses import replace

PREFIX = b'SHoEbox-EXP0305-diverse-record-context-v7-spacer\0'

def spacer(group_key, episode_index, record_position, length):
    if not isinstance(group_key, bytes) or not 0 <= episode_index < 2**64:
        raise ValueError('invalid group/cursor')
    if record_position not in (0, 1) or not 0 <= length <= 7:
        raise ValueError('invalid registered spacer')
    output = bytearray()
    block = 0
    while len(output) < length:
        digest = hashlib.sha256(PREFIX + group_key + b'\0'
            + episode_index.to_bytes(8, 'little') + bytes([record_position])
            + block.to_bytes(4, 'little')).digest()
        for byte in digest:
            if byte < 192:
                output.append(160 + byte % 96)
                if len(output) == length:
                    break
        block += 1
    return bytes(output)

def augment(ep, world, episode_index, offset, lengths):
    """Alternate 64-episode PAD/diverse blocks; preserve every other field."""
    if (episode_index // 64) % 2 == 0:
        return ep, []
    raw = bytearray(ep.data)
    positions = []
    at = 2 + offset
    for physical, length in enumerate(lengths):
        at += 5
        assert raw[at:at+length] == bytes([9]) * length
        raw[at:at+length] = spacer(world.group_key, episode_index, physical, length)
        positions.extend(range(at, at+length))
        at += length
    return replace(ep, data=bytes(raw)), positions
