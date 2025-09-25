# A basic set of PyTorch mechanics, illustrated through exemplars.
import torch

print("\n======== scalars =========")
national_debt = torch.tensor(3.7e13)
this_years_deficit = torch.tensor(1.8e12)
print(f"The national debt is currently ${national_debt.item()}.")
print(f"Next year it will be about ${national_debt + this_years_deficit}.")
print(f"The type is {national_debt.dtype}.")
print(f"The tensor type is: {type(national_debt)}.")
print(f"The tensor dtype is: {type(national_debt.dtype)}.")
print(f"The type of the item is: {type(national_debt.item())}.")
print(f"The tensor's shape is: {national_debt.shape}.")


print("\n======== vectors =========")
umw_loc = torch.tensor([38.3032, -77.4605])
print(f"UMW is at long/lat {umw_loc}.")
print(f"That tensor has shape {umw_loc.shape}.")
fifteen_thirteens = torch.zeros((15)) + 13
print(f"Here's a tensor of fifteen 13's: {fifteen_thirteens}.")
print(f"It has shape {fifteen_thirteens.shape}.")
elections = torch.arange(1788, 2028, 4)
print(f"Here's a tensor of the U.S. presidential election years:")
print(elections)
print(f"It has shape {elections.shape}.")
print(f"Here's the 'aftermath of a U.S. presidential election' years:")
aftermaths = elections + 1
print(aftermaths)
north_by_northwest_50_miles = torch.tensor([0.6668, -.3599])
new_place = umw_loc + north_by_northwest_50_miles
print(f"If I drove exactly 50 miles NNW from UMW, I'd be at {new_place}.")
nnw_200_miles = north_by_northwest_50_miles * 4
new_place = umw_loc + nnw_200_miles
print(f"And if I did 200 miles instead, I'd be at {new_place}.")

jezebel = torch.tensor([5.,9,1,1,2])  # Jezebel's answers to 5 survey questions
filbert = torch.tensor([4.,10,5,0,0]) # Filbert's answers to 5 survey questions
wendell = torch.tensor([8.,8,9,9,5])  # Wendell's answers to 5 survey questions
print(f"\nOur three eligible daters are {jezebel}, {filbert}, and {wendell}.")
jezebel = jezebel / jezebel.norm(2)
filbert = filbert / filbert.norm(2)
wendell = wendell / wendell.norm(2)
print(f"After (Euclidean) normalizing: {jezebel}, {filbert}, and {wendell}.")

print(f"\nAre Jezebel and Filbert an item? They get {jezebel @ filbert}")
print(f"What about Jezebel and Wendell? They get {jezebel @ wendell}")
print(f"Concluion: FILBERT gets the girl!")

print(f"\nJezebel currently has shape {jezebel.shape}.")
print(f"If we want room to expand to a whole army of Jezebels, we could unsqueeze her:")
print(jezebel.unsqueeze(0))
print(f"The Jezebels would then have shape {jezebel.unsqueeze(0).shape}")


print("\n======== matrices =========")
X = torch.tensor([[4.5, 5.5],[2.2,3.1],[-1.3,6.6]])
print(f"Here's X:")
print(X)
print(f"It has shape {X.shape}")

print(f"\nHere's X.T:")
print(X.T)
print(f"It has shape {X.T.shape}")

print(f"\nLet's get row 1, column 0 of X: {X[1,0]}")
print(f"Let's get row 1, column 0 of X.T: {X.T[1,0]}")
print(f"\nWe can always multiply a matrix by its transpose, of course:")
print(f"X @ X.T is:")
print(X @ X.T)
print(f"\nAnd we can always multiply a transpose of a matrix by it, too:")
print(f"X.T @ X is:")
print(X.T @ X)


print("\n======== tensors =========")
T = torch.tensor([
    [
        [
            [ 4.1, 5.6, 3.3, 1.1, 2.2 ],
            [ 2.0, 2.2, 8.0, 1.3, 1.8 ],
        ],
        [
            [ 1.1, 3.3, 6.6, 0.2, 0.3 ],
            [ 2.2, 4.4, 5.5, 0.5, 1.0 ],
        ],
        [
            [ 2.8, 3.9, 0.9, 3.0, 4.0 ],
            [ 3.0, 1.3, 5.3, 2.2, 1.3 ],
        ]
    ],
    [
        [
            [ 2.1, 0.2, 0.3, 0.5, 4.3 ],
            [ 8.0, 0.0, 0.0, 1.2, 2.1 ],
        ],
        [
            [ 5.9, 4.1, 9.9, 9.7, 1.2 ],
            [ 3.3, 1.3, 5.0, 5.6, 3.4 ],
        ],
        [
            [ 1.1, 2.2, 3.3, 4.4, 5.5 ],
            [ 9.9, 8.8, 7.7, 5.5, 6.6 ],
        ]
    ]
])

print("\nCheck out this lovely tensor T:")
print(T)
print(f"This tensor is shape {T.shape}")
print(f"How can we get element 1, 0, 0, 2 out of it?")
print(f"T[1,0,0,2] = {T[1,0,0,2]}: that's how!")

print("\nAnd now let's say we transpose it:")
print(T.T)
print(f"it becomes shape {T.T.shape}")
print("\nGenerally better is to specify explicitly which axes you want to swap, like so:")
transposed = T.transpose(3,1)
print("Transposing dimensions 3 and 1 gives us a tensor of shape {transposed.shape}.")
print(T.T)
print(f"it becomes shape {transposed.shape}")

# tensors
#   define
#   .shape
#   accessing elements
#   slices
#   .cat() and .stack()
#   .transpose()
#   .permute()
#   .clone()

# reductions (mean, sum, max, etc)
