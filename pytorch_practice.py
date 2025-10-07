# PyTorch practice worksheet (up to +20)
# Instructions: fill in the "STUDENT CODE" sections by carefully reading and
#   following the instructions.
# If any requested operations are impossible, set the corresponding variable to
#   the exact string "u cant do dat" instead.
# Then run the script to see which items are correct!
import torch

torch.manual_seed(0)

score = 0

# ---------------------------
# Item 1: Matrix multiplication
# Given the matrices x and y below, compute xy and yx using matrix
# multipliation. Put the answers in variables x_mm_y and y_mm_x, respectively.
# (Be sure to read the sentence above that starts with "If".)
x = torch.tensor([[5.0, 3.0],
                  [2.0, 2.5],
                  [1.0, 9.0]])
y = torch.tensor([[2.0, 2.0],
                  [3.0, 4.0]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 2: Scalar/vector ops & addition
# Compute the values 2v and v+w, for v and w tensors below. Put the answers in
# variables called v2 and v_plus_w.
v = torch.tensor([1., 2., 3.])
w = torch.tensor([4., 5., 6.])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 3: Dot product
# Compute the dot product of v and w, and also the dot product of w and v. Take
# a moment and ask yourself: what should the answers approximately be? Which
# answer should be larger? Put the answers in dot_vw and dot_wv. 
# === YOUR CODE GOES HERE ===

# ---------------------------
# Item 4: Norms
# Compute the Euclidean norm and the Manhattan norm of the v vector, and store
# those results in variables called v_l2 and v_l1, respectively. Ask yourself:
# which should be larger?
# Then normalize the vector v (using the Euclidean norm) and put the normalized
# result in the variable v_normalized. Ask yourself: what should the Euclidean
# norm of this normalized vector be? Compute it and see if you're right.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 5: Euclidean distance
# Compute the Euclidean ("crow flies") distance between the points at the
# arrow-ends of vectors v and w and put it in a variable called dist_vw.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 6: Cosine similarity
# Compute the cosine similarity between the vectors v and w, and store the
# answer in a variable called cos_vw.
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 7: Transpose
# Set a variable called M_T to the transpose of the matrix below. Then set a
# variable called M_T_T to that transpose transposed again. Ask yourself, what
# should M_T_T look like? Print it out and see if you're right.
M = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 8: Hadamard (element-wise) prod and Gram matrices.
# Set a variable A_square_elem to a 2x2 matrix that has each item of A, but
# squared. (Hint: this is super super easy to do in one operation. Don't treat
# each element as its own separate thing, square each, and then reassemble the
# matrix. My answer was a total of three characters long.) Then set variables
# A_mm_AT and AT_mm_A to be A times its transpose,
# and A-transpose times A, respectively. Ask yourself: should A_mm_AT and
# AT_mm_A have the same value? Print them out and see if you're right.
A = torch.tensor([[1., 2.],
                  [3., 4.]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 9: Row/col sums
# Given the matrix below, set variables row_sums and col_sums to be vectors
# with the sums of each row, and the sum of each column, respectively. Ask
# yourself: what shape/dimension should each of those variables be? Print them
# out and see if you're right.
C = torch.tensor([[1., 2.],
                  [3., 4.],
                  [5., 6.]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 10: Indexing & slicing
# Continuing to use the matrix C, from above, set a variable called first_row
# to be a vector that is the first row of C, and one called second_col to be a
# vector holding its second column. (Both operations should be really short and
# sweet, and should not involve you re-creating a vector from scratch with the
# desired contents.)
# === YOUR CODE GOES HERE ===

# ---------------------------
# Item 11: Reshape & flatten
# Given the vector "a" below, create a 3x4 matrix called M34 with the same 12
# elements rearranged into three rows. Then, use M34 to create a "flattened"
# version of M34 that has all the elements in a single vector again, and store
# that in a variable called a_flat. (The code should be simple and short.)
a = torch.arange(12.0)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 12: Squeeze & unsqueeze
# Examine the tensor q below, including its shape. Set its shape to a tensor
# called q_shape. Then create a tensor "simple" that has the same information
# but is of shape [2,3,2] (i.e., it removes the unnecessary, 1-dimensional
# axes). Then create a tensor "complicated" that has the same information but
# is of shape [1,2,1,1,3,2,1] (i.e., it inserts a couple of additional,
# unnecessary, 1-dimensional axes). 
q = torch.tensor([[[[[1.],[2.]], [[2.],[3.]], [[3.],[4.]]],[[[1.],[2.]], [[2.],[3.]], [[3.],[4.]]]]])
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 13: Stack & cat
# Given vectors u1 and u2, create a 2x3 matrix called stack_uv that has u1 on
# top of u2. Then create a 6-dimensional vector called cat_uv that has u1's
# contents followed by u2's.
u1 = torch.tensor([1., 2., 3.])
u2 = torch.tensor([4., 5., 6.])
# === YOUR CODE GOES HERE ===



# ---------------------------
# Item 14: Reductions & means
# Continuing with the C matrix defined above, set three variables sum_C, max_C,
# and mean_C that have scalars with the sum, max, and mean of all C's elements.
# (The code should be simple and short.)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 15: Batched matrix-mult
# Print out and consider the tensors X and Y below. Ask yourself: what shapes
# are X and Y? Should X @ Y be allowed? Should Y @ X? If so, what is shape of
# each? Put your answers to these in variables called X_dot_Y_shape and
# Y_dot_X_shape.
X = torch.stack([C, C + 1], dim=0)
D = torch.tensor([[7., 8., 9., 10.],
                  [10., 11., 12., 13.]])
Y = torch.stack([D, D], dim=0)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 16: Scalars (rank-0)
# Compute the sum, and the product, of the two scalar tensors below, then use
# .item() to get the actual float values. Save those float values in variables
# called a_plus_b and a_times_b.
a0 = torch.tensor(3.0)
b0 = torch.tensor(2.0)
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 17: Exponentials
# Make a 1-d tensor called "values" with the following values: -10, -1, 0,
# .001, 1, e, and 10. (For "e" you can use the constant "torch.e") Now, take a
# moment and ask yourself: "approximately what should I expect to get if I take
# "e-to-the" each one of those values?"
# Then, actually compute "e-to-the" that tensor, and store it in a variable
# called e_to_the. (In other words, the first element should be
# e-to-the-negative-10, the second element should be e-to-the-negative-1, etc.
# This can be done in one line, with no loop required.) Print it out for
# yourself. Were your guesses right?
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 18: Logs
# You still have your 1-d tensor called "values" from above. 
# Now, take a # moment and ask yourself: "approximately what should I expect to
# get if I take the (natural) log of each one of those values? Can I even *do*
# that for all of those values? And if not, which ones can't I do it for?"
# Then, actually compute the log of all values in that tensor (whether or not
# you think you can do them; PyTorch will helpfully give an answer of NaN for
# anything undefined), and store it in a variable called logs. (In other words,
# the first element should be the natural log of -10, the second element should
# be the log of -1, etc. This can be done in one line, with no loop required.)
# Print it out for yourself. Were your guesses right?
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 19: Sigmoid function
# You still have your 1-d tensor called "values" from above. 
# Now, take a # moment and ask yourself: "approximately what should I expect to
# get if I ran the sigmoid function on each one of those values?"
# Then, actually compute the sigma of all values in that tensor, and store it
# in a variable called sigmoids. (In other words, the first element should be
# the sigmoid function applied to the number -10, etc. I recommend writing a
# function called sigmoid() that takes a tensor argument and returns a tensor
# result, then call it to produce your answer with one line of code.)
# Print it out for yourself. Were your guesses right?
# === YOUR CODE GOES HERE ===


# ---------------------------
# Item 20: One neuron
#
# Define a function called "jezebel_neuron()" that will compute the probability
# that Jezebel will be attracted to a particular romantic partner. It should
# take a rank-1, 5-dimensional tensor as input, and return a rank-1,
# 5-dimensional tensor as output. The output should be that of a single neuron
# (put another way, a single logistic regression) with the weights equal to (in
# order): 5, -1.2, -2, 0, 6.
# Then execute this function on the three variables filbert, wendell, and biff,
# defined below. Before doing so, ask yourself: what range would I expect the
# answer to be in? And would I expect the answer to be higher for filbert,
# wendell, or biff? Then print out your answers and see if you're right. Store
# your answers in variables called filbert_prob, wendell_prob, and biff_prob.
# Finally, stack the three boys in a single matrix (to make a rank-2 tensor of
# shape 3x5) called "boys". Filbert should be on row 1, wendell on row 2, and
# biff on row 3. Then run your jezebel_neuron() function on it all in one go,
# and store the answers in a rank-2, 3x1 tensor boy_probs.

filbert = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
wendell = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
biff = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])

# === YOUR CODE GOES HERE ===











# =====================================================
# ================== CHECKER SCRIPT ===================
def exists_all(names: list) -> bool:
    g = globals()
    return all(name in g for name in names)

def check_item(idx: int, needed: list, predicate):
    global score
    if not exists_all(needed):
        print(f"(Item #{idx} incomplete.)")
        return
    try:
        ok = predicate()
    except Exception:
        ok = False
    if ok:
        print(f"Item #{idx} correct!")
        score += 1
    else:
        print(f"Item #{idx} INCORRECT!")

def _item1_expected():
    exp_xy = torch.tensor([[19.0000, 22.0000],
                           [11.5000, 14.0000],
                           [29.0000, 38.0000]])
    exp_yx = "u cant do dat"
    return exp_xy, exp_yx

# Run all checks
check_item(1, ["x_mm_y", "y_mm_x"], lambda: (
    isinstance(x_mm_y, torch.Tensor) and torch.allclose(x_mm_y, _item1_expected()[0], atol=1e-4)
    and isinstance(y_mm_x, str) and y_mm_x == _item1_expected()[1]
))
check_item(2, ["v2", "v_plus_w"], lambda: (
    torch.allclose(v2, torch.tensor([2., 4., 6.])) and
    torch.allclose(v_plus_w, torch.tensor([5., 7., 9.]))
))
check_item(3, ["dot_vw","dot_wv"], lambda: float(dot_vw) == float(torch.dot(v,
w)) and float(dot_wv) == float(torch.dot(v, w)))
check_item(4, ["v_l2","v_l1","v_normalized"], lambda: torch.isclose(torch.tensor(float(v_l2)),
    torch.linalg.norm(v), atol=1e-5) and torch.isclose(torch.tensor(float(v_l1)),
    torch.linalg.norm(v,1), atol=1e-5) and
    torch.isclose(v_normalized, v/v_l2).all())
check_item(5, ["dist_vw"], lambda: torch.isclose(torch.tensor(float(dist_vw)), torch.dist(v, w), atol=1e-5))
check_item(6, ["cos_vw"], lambda: torch.isclose(torch.tensor(float(cos_vw)), torch.nn.functional.cosine_similarity(v, w, dim=0), atol=1e-6))
check_item(7, ["M_T","M_T_T"], lambda: torch.equal(M_T, M.t()) and
    torch.equal(M,M_T_T))
check_item(8, ["A_square_elem", "A_mm_AT","AT_mm_A"], lambda: (
    torch.equal(A_square_elem, A * A) and torch.equal(A_mm_AT, A @ A.t())
and torch.equal(AT_mm_A, A.T @ A)
))
check_item(9, ["row_sums", "col_sums"], lambda: (
    torch.equal(row_sums, C.sum(dim=0)) and torch.equal(col_sums, C.sum(dim=1))
))
check_item(10, ["first_row", "second_col"], lambda: (
    torch.equal(first_row, C[0]) and torch.equal(second_col, C[:, 1])
))
check_item(11, ["M34", "a_flat"], lambda: (
    torch.equal(M34, a.reshape(3, 4)) and torch.equal(a_flat, a.reshape(3, 4).flatten())
))
check_item(12, ["q_shape", "simple", "complicated"], lambda: (
    list(q_shape) == [1,2,3,2,1] and
    list(simple.shape) == [2,3,2] and
    list(complicated.shape) == [1,2,1,1,3,2,1] and
    torch.equal(q.flatten(), simple.flatten()) and
    torch.equal(complicated.flatten(), simple.flatten())
))
check_item(13, ["stack_uv", "cat_uv"], lambda: (
    torch.equal(stack_uv, torch.stack([u1, u2], dim=0)) and
    torch.equal(cat_uv, torch.cat([u1, u2], dim=0))
))
check_item(14, ["sum_C", "mean_C", "max_C"], lambda: (
    torch.equal(sum_C, C.sum()) and
    torch.equal(max_C, C.max()) and
    torch.equal(mean_C, C.mean())
))
check_item(15, ["X_dot_Y_shape", "Y_dot_X_shape"], lambda: (
    isinstance(X_dot_Y_shape, list) and X_dot_Y_shape == [2, 3, 4] and
    isinstance(Y_dot_X_shape, str) and Y_dot_X_shape == "u cant do dat"
))
check_item(16, ["a_plus_b", "a_times_b"], lambda: (
    float(a_plus_b) == float(a0 + b0) and float(a_times_b) == float(a0 * b0)
))
check_item(17, ["e_to_the"], lambda: (
    torch.isclose(e_to_the,torch.tensor([0.0000,0.3679,1.0000,1.0010,2.7183,15.1543,22026.4648]), atol=1e-4).all()
))
check_item(18, ["logs"], lambda: (
    torch.isclose(logs[3:],torch.tensor([-6.9078,0.0000,1.0000,2.3026]),
atol=1e-4).all()
))
check_item(19, ["sigmoids"], lambda: (
    torch.isclose(sigmoids,torch.tensor([0.0000,0.2689,0.5000,0.5002,0.7311,0.9381,
        1.0000]), atol=1e-4).all()
))
check_item(20, ["boy_probs","filbert_prob","wendell_prob","biff_prob"], lambda: (
    torch.isclose(filbert_prob,torch.tensor([1.0]), atol=1e-4).all() and
    torch.isclose(wendell_prob,torch.tensor([0.0392]), atol=1e-4).all() and
    torch.isclose(biff_prob,torch.tensor([0.9526]), atol=1e-4).all() and
    torch.isclose(boy_probs,torch.tensor([1.0000,0.0392,0.9526]), atol=1e-4).all()
))

print(f"You got +{score}XP! (out of a possible 20XP)")
