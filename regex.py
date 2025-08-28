'''
Allow the user to interactively search a corpus (Bob Dylan's lyrics, e.g.)
for regex matches.

DATA 470D3
Stephen Davies, University of Mary Washington, fall 2025
'''
import re

CORPUS = "dylan.txt"

with open(CORPUS, encoding="utf-8") as f:
    lines = [ l.strip() for l in f.readlines() ]

regex = input("Enter a regex (or 'done'): ")
tot_matches = 0
while regex != "done":
    for line in lines:
        list_of_matches = re.findall(regex, line)
        if list_of_matches:
            print("\n" + line)
            for match in list_of_matches:
                print("\tmatch: " + str(match))
                tot_matches += 1
    print(f"\nThere were {tot_matches} matches in the corpus.\n")
    regex = input("Enter a regex (or 'done'): ")

