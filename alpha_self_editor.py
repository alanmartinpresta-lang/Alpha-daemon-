
"""
Alpha True Self-Edit Prototype

Alpha can inspect its whole project inside an isolated workspace and create
candidate versions of itself. The evaluator and substrate are deliberately
outside this module.

This module does not pretend to be a general LLM. It provides the actual
mechanics required for Alpha to:
  1. inspect project source;
  2. identify editable Python functions/constants;
  3. create a candidate patch;
  4. compile the candidate;
  5. record the diff;
  6. request external evaluation;
  7. adopt or rollback.

The policy deciding WHY to edit remains replaceable. That separation lets us
later connect a stronger reasoning model without changing the safety boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import ast, difflib, json, shutil, tempfile, py_compile

@dataclass
class Candidate:
    path: str
    reason: str
    old_source: str
    new_source: str
    diff: str

class SelfEditor:
    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        self.backup = self.root.parent / (self.root.name + "_parent")
        self.backup.mkdir(exist_ok=True)

    def source_files(self):
        return [p for p in self.root.rglob("*.py")
                if "__pycache__" not in p.parts and ".git" not in p.parts]

    def inventory(self):
        out=[]
        for p in self.source_files():
            text=p.read_text(encoding="utf-8")
            try:
                tree=ast.parse(text)
            except SyntaxError:
                continue
            funcs=[n.name for n in ast.walk(tree)
                   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
            classes=[n.name for n in ast.walk(tree)
                     if isinstance(n,ast.ClassDef)]
            out.append({
                "path":str(p.relative_to(self.root)),
                "lines":len(text.splitlines()),
                "functions":funcs,
                "classes":classes,
            })
        return out

    def propose_constant_mutation(self, path, lineno, factor, reason):
        p=self.root/path
        old=p.read_text(encoding="utf-8")
        lines=old.splitlines()
        line=lines[lineno-1]
        tree=ast.parse(old)
        target=None
        for n in ast.walk(tree):
            if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) \
               and not isinstance(n.value,bool) and n.lineno==lineno:
                target=n; break
        if target is None:
            raise ValueError("No editable numeric constant on target line")
        v=float(target.value)
        nv=v*factor
        token=str(int(v)) if v.is_integer() else repr(v)
        repl=str(int(round(nv))) if v.is_integer() else repr(nv)
        # conservative exact-token replacement
        import re
        new_line,n=re.subn(r'(?<![\w.])'+re.escape(token)+r'(?![\w.])',
                           repl,line,count=1)
        if n!=1:
            raise ValueError("Could not produce unambiguous patch")
        lines[lineno-1]=new_line
        new="\n".join(lines)+"\n"
        diff="".join(difflib.unified_diff(
            old.splitlines(True),new.splitlines(True),
            fromfile=str(path),tofile=str(path)))
        return Candidate(str(path),reason,old,new,diff)

    def compile_candidate(self, cand: Candidate):
        # Compile in a temporary file without changing the live parent.
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/Path(cand.path).name
            p.write_text(cand.new_source,encoding="utf-8")
            py_compile.compile(str(p),doraise=True)
        return True

    def adopt(self, cand: Candidate):
        p=self.root/cand.path
        p.write_text(cand.new_source,encoding="utf-8")

    def rollback(self, cand: Candidate):
        (self.root/cand.path).write_text(cand.old_source,encoding="utf-8")

    def save_manifest(self, path, data):
        (self.root/path).write_text(json.dumps(data,indent=2),encoding="utf-8")
