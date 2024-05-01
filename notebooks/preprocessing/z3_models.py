import z3aux
from z3 import *

s = Solver() # Z3 solver instance
x,y = Ints('x y') # Int-typed variables

# x := avg_ep_dur, y := num_eps
# (x > 1800 | (y >= 6 & x >= 240)) & (x > 1800 | y < 6) & (x <= 1800)

# Add constraints
s.assume(x > 1800)
s.assume(Or(y >= 6, x >= 240))
s.assume(Or(x > 1800, y < 6))
s.assume(x <= 1800)

# Print all models
result = s.check()
while result == sat:
  m = s.model()
  print("Model: ", m)
  
  v_x = m.eval(x, model_completion=True)
  v_y = m.eval(y, model_completion=True)

  s.add(x != v_x, y != v_y)
  result = s.check()

print(result, "--> no further models")
