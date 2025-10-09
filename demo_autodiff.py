# PyTorch's autodiff (which they called "autograd") in action through an
# example.
import torch

# We're going to construct a function z that has two inputs: 
#    - A 3x1 tensor x
#    - A 1x1 tensor y
#
# Our function will be:
# z  =  (Σ x^2) + y^3  =  x_0^2 + x_1^2 + x_2^2 + y^3
#
# We seek ∂z/∂x (with three components), and ∂z/∂y (one component).
#
# Analytically, from calculus, we know:
#     ∂z/∂x = [ 2x_0, 2x_1, 2x_2 ]
#     ∂z/∂y = 3y^2
#
# Let's evaluate those partial derivatives at the values of x and y listed
# below, and wrap our head around what they mean.

# Create our inputs x and y. The "requires_grad=True" thing means "we're going
# to be using autodiff, so please stand by for that, PyTorch."
x = torch.tensor([1.,2.,3.], requires_grad=True)
y = torch.tensor([2.], requires_grad=True)

# Create some other intermediate functions to represent components of the
# calculation.
q = x ** 2
s = q.sum()
r = y ** 3

# Finally, compute our actual answer:
z = s + r

# And now, tell PyTorch "we'd like the partial derivatives you've been tracking
# via autodiff, PyTorch, so please give them to us now." We do this by calling
# .backward() on our output, and then examining x.grad and y.grad.
z.backward()

print(f"x = {x}")
print(f"q = {q}")
print(f"s = {s}\n")
print(f"y = {y}")
print(f"r = {r}\n")
print(f"Final answer: z = {z}\n")

print("And now, the partial derivatives (evaluated at the current x and y):")
print(f"x.grad, a.k.a. ∂z/∂x = {x.grad}")
print(f"y.grad, a.k.a. ∂z/∂y = {y.grad}")

# According to these outputs, then, z changes by a factor of 2 for every
# unit smidge that x_0 is moved up or down from where it currently is.
#
# Also, z changes by a factor of 4 for every unit smidge that x_1 is moved up
# or down from where it currently is.
#
# Also, z changes by a factor of 6 for every unit smidge that x_2 is moved up
# or down from where it currently is.
#
# And finally, z changes by a factor of 12 for every unit smidge that y is
# moved up or down from where it currently is.
#
# (Try it! Change one of x_0, x_1, x_2, or y, tweak it up or down a smidge, and
# see how much z changes as a result.)
#
# Now play with the formula. (For instance, change sum() to min() or max() and
# see what happens.)
