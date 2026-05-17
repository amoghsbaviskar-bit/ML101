## Day - 06: Dataset and Dataloader Introduction

## What I did:
* Started storing x and y tensor in a class instead of keeping them open
* Used dataloader for feeding the values in batches rather than all at once also used shuffle to randomize the data at every epoch

---

## Output:
* 'loss.png': shows the loss graph
* 'decision_boundary': shows what model understands 
* 'output.txt': shows the terminal output 

---

## Observations:
* The loss curve was much steeper this time than the one in day 05
  it had a slow decent around loss 0.2
* Sigmoid curve was similar to that of day 05
* The test value leaned towards PASS

---

## Learning:
* Learned the use of torch.utils.data in it Datasets and Dataloader function 
* Learned how nested loop worked in this case : outer loops handels eopches while inner loops deals with every batch individually

---