# Day 03 - Refactoring Quadratic regression model using torch.nn and torch.optim

## What I did:

* Took the existing quadratic regression model and used torch.nn and torch.optim 
* Instead of having my parameters open in the code I made a class called quadmodel to store them 
* In class quadmodel I also wrote a function called forward which helped me in combining the formula in one spot 
* I used criterion = nn.MSELoss() instead of manually typing the formula 
* I used optimizer = optim.Adam(model.parameters(), lr = 0.1) to define my learning rate than keeping it open in the program 
* optimizer.zero_grad(),loss.backward(),optimizer.step() these helped me wipe the old math , calculate direction to move and update my weights respectively
* Lastly I used if epoch % 50 == 0:
        print(f"Epoch:{epoch},Loss:{loss.item()}") to avoid printing alot of results and keep it compact in the terminal 

---

## Model used: 
* y = w₁x² + w₂x + b

---

## Output:
* 'output.txt' it shows all the epoch and loss which were in the terminal
* 'loss.png' it shows the training loss curve over epochs

---

## Learning:
* I learned how to use class to define the model and organize it 
* Applied torch.nn and torch.optim 
* Learned to keep the terminal from getting messy 

---
