## Day - 05: My First Classifier

## What i did:
* Made utils folder 
* added plotting.py in utils folder 
* plotting.py saves me from re writting 
mlt.plot(losses)
mlt.xlabel("Epoch")
mlt.ylabel("Loss")
mlt.title("Training Loss Curve")

mlt.savefig("loss.png")

mlt.show() this every time in the prog 
* also added a new function in plotting.py to give me a decision plot 
* Switched from MSEloss to BCEloss
* Started using more data for x and y than previous days 
* Instead of defining new layer for each started using nn.Sequential()
* Used nn.Sigmoid to get output in 1 and 0 
* Tested the model with a test_val [[5.0]] as it was not in the training data 
* The result came to be PASS

--- 
## Output:
* 'loss.png': shows the loss graph
* 'decision_boundary': shows what model understands 
* 'output.txt': shows the terminal output 

---
## Observation: 
* as seen in day04 loss graph it was just a bump today however the loss graph has a slow desent to accuracy this happend because sigmoid fuction let it have a slow descent while the initial steep was from bceloss
* I tested the 0.5 value it sat right between the fail numbers 1.0 to 4.0 and pass numbers 6.0 to 9.0 my model leaned the 0.5 as a pass value
* Initially the utils folder was not being located by the program so i added 
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root)) which solved the issue

--- Learning:
* linear regression shows us how much while what I did today
with sigmoid showed me yes/no which is logistic regression
* MSEloss didnt penalize the model enough for confident wrong guess BCEloss however aggressively penalizes and forces faster convergence

---