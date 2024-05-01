from z3 import *

# A few convenience methods

def assume(self, *assumptions):
  self.add(assumptions)

def prove(self, *goals):
  self.add(Not(And(goals)))
  result = self.check()
  if result == unsat:
    print("OK")
  elif result == unknown:
    print("UNKNOWN")
  else:
    m = self.model()
    print("FAIL")
    print(m)

def solve(self, tag, *constraints):
  result = self.check(constraints)
  if result == sat:
    m = self.model()
    print(f"{tag}: SAT")
    print(m)
  elif result == unsat:
    print(f"{tag}: UNSAT")
  else:
    print("UNKNOWN")

Solver.assume = assume
Solver.prove = prove
Solver.solve = solve
