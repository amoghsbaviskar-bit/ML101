# Day 02 - Quadratic Regression from Scratch

## What I did

* Extended linear regression to a quadratic model using PyTorch  
* Built a custom polynomial model using \(x^2\), \(x\), and bias  
* Trained the model using manual gradient descent  
* Tuned learning rate to prevent divergence (NaN issues)  
* Logged loss values over epochs  
* Observed stability vs explosion behavior during training  

---

## Model Used

y = w₁x² + w₂x + b

---

## Output

* `loss.png` → shows training loss curve over epochs  
* `output.txt` → contains epoch-wise training logs  

---

## Learning

* Understood feature engineering using \(x^2\)  
* Learned why learning rate affects stability (exploding vs converging gradients)  
* Saw how polynomial models extend linear regression  
* Experienced gradient explosion → inf → nan and fixed it by tuning LR  
* Understood that loss convergence depends on both model + optimization stability  
* Strengthened understanding of full training loop (forward → loss → backward → update)  

---

## Key Insight

* Linear regression learns a straight line  
* Quadratic regression learns curvature  
* Stability matters as much as correctness in ML training  