# csc413-2516-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [CSC413-2516 Homework 3 Solved](https://www.ankitcodinghub.com/product/csc413-2516-submission-you-must-submit-your-solutions-as-a-pdf-file-through-markus-you-can-produce-the-file-however-you-like-e-g-latex-microsoft-word-scanner-as-long-as-it-is-readable-solv/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;118063&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413-2516 Homework 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
Submission: You must submit your solutions as a PDF file through MarkUs . You can produce the file however you like (e.g. LaTeX, Microsoft Word, scanner), as long as it is readable.

1 Trading off Resources in Neural Net Training

1.1 Effect of batch size

When training neural networks, it is important to select appropriate learning hyperparameters such as learning rate, batch size, and the optimizer for training. In this question, we will investigate the effect of how varying these quantities will affect the validation loss during training.

1.1.1 Batch size vs. learning rate

For simplicity, we only consider the scalar version of the NQM. We have the quadratic loss

, where a &gt; 0 and w ∈ R is the weight that we would like to optimize. Assume that

we only have access to a noisy version of the gradient — each time when we make a query for the gradient, we obtain g(w), which is the true gradient ∇L(w) with additive Guassian noise:

g(w) = ∇L(w) + ϵ, ϵ ∼ N(0,σ2).

SGD updates: w ← w − ηg(w).

One way to reduce noise in the gradient is to use minibatch training. Let B be the batch size, and denote the minibatch gradient as gB(w):

, where gi(w) = ∇L(w) + ϵi, ϵi i.i.d.∼ N(0,σ2).

Mini-batch SGD updates: w ← w − ηgB(w).

[0.5pt] As batch size increases, how do you expect the optimal learning rate to change? Briefly explain in 2-3 sentences.

(Hint: Think about how the minibatch gradient noise change with B.)

1.1.2 Training steps vs. batch size

For most of neural network training in the real-world applications, we often observe the relationship of training steps and batch size for reaching a certain validation loss as illustrated in Figure 1. Such a relationship can be observed across various architectures.

(a) [0.5pt] For the three points (A,B,C) on Figure 1, which one has the most efficient batch size (in terms of best resource and training time trade-off)? Assume that you have access to scalable (but not free) compute such that minibatches are parallelized efficiently. Briefly explain in 1-2 sentences.

(b) [0.5pt] Figure 1 demonstrates that there are often two regimes in neural network training: the noise dominated regime and the curvature dominated regime. In the noise dominated regime, the bottleneck for optimization is that there exists a large amount of gradient noise. In the curvature dominated regime, the bottleneck of optimization is the ill-conditioned loss landscape. For points A and B on Figure 1, which regimes do they belong to? Fill each of the blanks with one best suited option. Options: noise dominated / curvature dominated.

Point A: Regime: . Point B: Regime: .

1.1.3 Batch size, Optimizer, Normalization, Learning Rate

I Use an adaptive, 2nd, or higher order optimizer

II (II+) Increase the batch size, (II-) Decrease the batch size

III (III+) Increase the learning rate, (III-) Decrease the learning rate

IV Use a normalization technique such as batch norm

(a) [1pt] Regime A is dominated by curvature in Figure 2. Which of the above options will accelerate

the training by addressing the ill-conditioned curvature.

(Select all that apply from I, II+, II-, III+, III-, IV)

(b) [1pt] Regime B is dominated by the noise in Figure 2. Which of the above options will accelerate

the training by addressing the large amount of gradient noise? (Select all that apply from I, II+, II-, III+, III-, IV)

1.2 Model size, dataset size and compute

We have seen in the previous section that batch size and learning rate are important hyperparameter to help with optimization. Besides efficiently minimizing the training loss, we are also interested in the test loss. Recently, researchers have observed an intriguing relationship between the test loss and hyperparameters such as the model size, dataset size and the amount of compute used.

Note: You can think of the total compute used, shown in x-axis, as number of training steps × number of parameters × constants.

(a) [1pt] There are two language models shown in Figure 3: Model A in green and Model B in purple. We also highlight an intersection point “X” that denotes a target test loss achieved by both models using the same total amount of compute. (1) Which of the two models has more parameters? Why? (2) At the intersection “X”, which of the two models has been training for more iterations/updates? Why?

Explain your answers in no more than 3 sentences.

(b) [1pt] Suppose you want to train a language model to reach a desired test loss.

2 Generalization Error of Linear Regression

wˆ = argmin . (2.1)

w∈Rd n i=1

Denote X = [x1;x2;…xn] ∈ Rn×d as the training data matrix (design matrix), and t ∈ Rn as the corresponding label vector. When X has full-rank, the solution of the above problem is given as

((X⊤X)−1X⊤t, n &gt; d,

wˆ = (2.2)

X⊤(XX⊤)−1t, n &lt; d.

We restrict ourselves to a linear teacher model with Gaussian features, i.e.,

, (2.3)

where εi is i.i.d. label noise with mean 0 and variance σ2, and w∗ ∈ Rd is the true signal that does not depend on x,ε. To further simplify the computation, we consider a Bayesian setting and place an isotropic prior on the true signal : E[w∗w∗⊤] = d1Id.

Our goal is to compute the generalization error (test loss) of the linear regression model:

, (2.4)

where the test data is drawn from the same distribution: x˜ ∼ N(0,Id).

2.1 Bias-variance decomposition

2.1.1 [0.5pt]

When n &gt; d (underparameterized regime), show that the test loss can be written as

. (2.5)

bias, B(wˆ) | {z } variance, V(wˆ)

Hint: use the i.i.d. property of the training data and label noise: E[xx⊤] = Id, E[εε⊤] = σ2In, where .

2.1.2 [0pt]

When n &lt; d (overparameterized regime), show that the test loss can be written as

. (2.6)

| {z } | {z }

bias, B(wˆ) variance, V(wˆ)

Hint: for the bias term, use the isotropic prior: .

2.2 Deriving the exact expressions

• When d &gt; n, Tr I .

• Given Gaussian random matrix G ∈ Rm×p, where [G]ij ∼ N(0,1), we have

, if m &gt; p + 1. (2.7)

Bonus: argue why this is the case, using properties of the χ2-distribution.

2.2.1 [1pt]

2.2.2 Double descent [1pt]

Answer based on the formulas: (1) under what conditions (on n,d,σ) is it possible for the model to achieve perfect generalization, i.e., R(wˆ) = 0? (2) Does adding more training examples always help generalization? Why? Briefly explain your answers in no more than 3 sentences.

2.2.3 [0pt]

Plot the expression you derived in the previous question with the following parameters: fix the number of features d = 500, and vary training set size n from 100 to 1000 (make sure you exclude the regime n − 1 ≤ d ≤ n + 1). Include your figures for σ = 0 (noiseless) and σ = 0.2 (noisy). Answer the following questions based on the figure:

• What difference do you observe between the noiseless and noisy setting?

• Does adding more training data (increasing n) always lead to better test performance? If not, under what conditions (on n,d,σ) is it beneficial? And when does it hurt?

2.3 Ridge regularization

To reduce overfitting to the label noise, we now consider regularizing the ℓ2-norm of the parameters. For λ &gt; 0, the regularized objective is given as

wˆ λ = argmin , (2.8)

w∈Rd n i=1

2.3.1 [0pt]

By taking the derivative of the objective (2.8) and setting it to zero , show that wˆ λ has the following closed-form:

wˆ λ = (X⊤X + nλId)−1X⊤t. (2.9)

2.3.2 [0.5pt]

Heuristically argue whether the regularization strength λ should increase or decrease with the training set size n and noise level σ. Provide your arguments in no more than two sentences.

2.3.3 [0pt]

Define and set λ = σ2γ (why does this choice of λ make sense?). Under the same setting as Section 2.1, show that the generalization error (2.4) can be simplified to

. (2.10)

2.3.4 [0.5pt]

Using this result, we can plot the expected risk ER(wˆ λ) for our choice of .

Figure 4: Generalization error of ridge regression. Similar to Section 2.2.3, we fix the number of features d = 500, and vary training set size n from 100 to 1000. We choose σ = 0.2.

Based on the figure, answer the following questions:

• Compare the test loss of the unregularized estimator E[R(wˆ)] in Section 2.2.1 against the ridge-regularized E[R(wˆ λ)]. What difference do you observe?

• Under appropriate ridge regularization (λ = σ2γ), does adding more training data (increasing

n) always lead to better test performance?

References

[Bai and Silverstein, ] Bai, Z. and Silverstein, J. W. Spectral analysis of large dimensional random matrices, volume 20. Springer.
