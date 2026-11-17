"""Correctness-only compiler wrapper: instrument *every project C++ function,
loop iteration, comparison, and automatic object declaration before execution.
Original files, model layout, checkpoints and timing libraries are untouched.
Per-site maps retain original text and offsets; zero counters are explicit.
"""
import hashlib,json,os,re,resource,subprocess,sys,time
from pathlib import Path
from tree_sitter import Language,Parser
import tree_sitter_cpp
PARSER=Parser(Language(tree_sitter_cpp.language()))

def transform(raw,name,sites):
    tree=PARSER.parse(raw);edits={}
    def add(pos,s):edits.setdefault(pos,[]).append(s)
    def event(node,kind,fn,amount='1'):
        i=len(sites);sites.append(dict(id=i,file=name,line=node.start_point.row+1,kind=kind,function=fn,text=raw[node.start_byte:node.end_byte].decode()[:1600]))
        return f'(::a11_event::counts[{i}]+={amount})'
    def text(n):return raw[n.start_byte:n.end_byte].decode()
    def var(n):
        while n and n.type not in ('identifier','structured_binding_declarator'):
            if n.type=='function_declarator':return None
            n=n.child_by_field_name('declarator')
        return text(n) if n and n.type=='identifier' else None
    def walk(n,fn=None):
        if n.type in ('static_assert_declaration','static_assert','type_definition'):return
        if n.type=='declaration' and re.search(r'\b(constexpr|consteval)\b',text(n)):return
        if n.type in ('function_definition','lambda_expression'):
            b=n.child_by_field_name('body')
            if not b:return
            prefix=raw[n.start_byte:b.start_byte].decode()
            if re.search(r'\b(constexpr|consteval)\b',prefix):return
            fn=prefix.strip();add(b.start_byte+1,event(n,'function',fn)+';')
            # Semantic events supplement the exhaustive source-site ledger.
            if re.search(r'\bModel::observe\(uint8_t byte\)',prefix):
                for kind,amount in [('input_bytes','1'),('padding_bytes','uint64_t(byte==9)'),('nuisance_bytes','uint64_t(byte>=160)')]:
                    add(b.start_byte+1,event(n,kind,fn,amount)+';')
            if re.search(r'\bModel::teach\(uint8_t target\)',prefix):
                add(b.start_byte+1,event(n,'native_target_calls',fn)+';')
                add(b.start_byte+1,event(n,'native_update_targets',fn,'uint64_t(training)')+';')
            if re.search(r'\baffine\(',prefix):add(b.start_byte+1,event(n,'affine_rows',fn)+';')
            if re.search(r'\ba(?:10|11)_affine\(',prefix):
                add(b.start_byte+1,event(n,'affine_rows',fn,'Rows')+';')
            if re.search(r'\baffine_step\(',prefix):add(b.start_byte+1,event(n,'local_gradient_rows',fn)+';')
        if fn:
            if n.type in ('for_statement','for_range_loop','while_statement','do_statement'):
                b=n.child_by_field_name('body')
                if b:
                    c=event(n,'iteration',fn)+';'
                    if b.type=='compound_statement':add(b.start_byte+1,c)
                    else:add(b.start_byte,'{'+c);add(b.end_byte,'}')
            if n.type=='binary_expression':
                op=n.child_by_field_name('operator')
                if op and text(op) in ('==','!=','<','>','<=','>='):
                    add(n.start_byte,'('+event(n,'comparison',fn)+',(');add(n.end_byte,'))')
            if n.type=='declaration' and n.parent.type=='compound_statement':
                # sizeof excludes storage owned indirectly by vectors/pointers.
                # References and static storage are not automatic allocations.
                decl=text(n)
                if not re.search(r'\b(static|extern|constexpr|thread_local|typedef)\b',decl):
                    names=[]
                    for i,c in enumerate(n.children):
                        if n.field_name_for_child(i)=='declarator' and '&' not in text(c.child_by_field_name('declarator') or c).split('=')[0]:
                            v=var(c)
                            if v:names.append(v)
                    if names:
                        add(n.end_byte,event(n,'automatic_object_activations',fn)+';'+event(n,'automatic_object_bytes',fn,'+'.join('sizeof('+v+')' for v in names))+';')
        for c in n.named_children:walk(c,fn)
    walk(tree.root_node)
    for pos in sorted(edits,reverse=True):raw=raw[:pos]+''.join(edits[pos]).encode()+raw[pos:]
    return raw

def main():
    args=sys.argv[1:];real=os.environ.get('A11_REAL_CXX','/usr/bin/g++')
    if not any(x.endswith(('.cpp','.cc','.cxx')) for x in args):
        os.execv(real,[real,*args])
    root=Path(os.environ['A11_INSTRUMENT_ROOT']);root.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();cpu=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    build=root/('build_'+str(os.getpid()));build.mkdir()
    includes=[Path(a[2:]).resolve() for a in args if a.startswith('-I')]
    sites=[];copied={};originals={}
    def copy(path):
        path=path.resolve()
        if path in copied:return copied[path]
        target=build/(hashlib.sha256(str(path).encode()).hexdigest()[:12]+'_'+path.name);copied[path]=target
        raw=path.read_bytes();originals[str(path)]=hashlib.sha256(raw).hexdigest()
        # Instrument before include rewrites, retaining original line/byte identity.
        data=transform(raw,str(path),sites)
        def inc(m):
            name=m.group(1).decode();opts=[path.parent/name,*[p/name for p in includes]]
            found=next((p for p in opts if p.is_file()),None)
            if found is None:raise FileNotFoundError((path,name))
            return b'#include "'+str(copy(found)).encode()+b'"'
        data=re.sub(rb'#include\s*"([^"\n]+)"',inc,data)
        target.write_bytes(data);return target
    commands=[]
    for a in args:commands.append(str(copy(Path(a))) if a.endswith(('.cpp','.cc','.cxx')) else a)
    # A distinct hidden counter table per DSO/executable; no serialized fields.
    h=build/'events.hpp'
    h.write_text('''#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
namespace a11_event {
inline __attribute__((visibility("hidden"))) uint64_t counts['''+str(len(sites))+''']{};
struct __attribute__((visibility("hidden"))) Logger {
~Logger(){const char* d=std::getenv("A11_EVENT_DIR");if(!d)return;char p[4096];
std::snprintf(p,sizeof(p),"%s/%d-'''+build.name+'''.bin",d,int(getpid()));
FILE* f=std::fopen(p,"wb");if(f){std::fwrite(counts,sizeof(counts),1,f);std::fclose(f);}}
};
inline __attribute__((visibility("hidden"))) Logger logger;
}
''')
    (build/'sites.json').write_text(json.dumps(sites,indent=2)+'\n')
    command=[real,'-include',str(h),*commands]
    rc=subprocess.run(command).returncode
    now=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
    (build/'compile.json').write_text(json.dumps(dict(command=command,original_command=args,exit_code=rc,source_hashes=originals,
      wall_seconds=time.perf_counter()-start,wrapper_cpu_seconds=now.ru_utime+now.ru_stime-cpu.ru_utime-cpu.ru_stime,
      compiler_cpu_seconds=ch.ru_utime+ch.ru_stime-child.ru_utime-child.ru_stime,
      counted_sites=len(sites)),indent=2)+'\n')
    sys.exit(rc)
if __name__=='__main__':main()
