## End-to-End Memory Networks on bAbI Dataset 

Full report and Figures: [*MemN2N_QA*](Final%20Report.pdf)

### Abstract
Attention has gained a lot of popularity due to its performance on text and image data. It is able to extract only the words or pixels that are related to the task. Applying this idea to memory has a great potential and variable use cases. In this report we will inquire how the original author of the paper "End-to-End Memory Networks" (Sukhbaatar, et al.) implemented memory; and we will mention different approaches for the model. We will explain the dataset we used for the tasks; then we will compare how our results stand out against the paper's result.

### Dataset
The ​paper uses the memory network on text data to process question-answering (QA) tasks. The tasks are provided by Facebook Research, and are "organized towards the goal of automatic text understanding and reasoning."(Weston, et al.) The 20 tasks promote different aspect of understanding. Some example tasks are, starting from the simpler tasks, single supporting fact (Figure 1), yes/no question (Figure 2), basic induction (Figure 3), and size reasoning (Figure 4).

### Results
Wide variety of models we tried gave wide variety of results. Implementing the MemNN model only gave similar results with the paper for the single supporting fact, however not good enough in the other tasks. We got 94% accuracy in task 1, while the paper suggests 98%. For the rest of the tasks we got the half of the accuracies that the paper claimed they did.

Though, when we look at the MemN2N model, which has the concept of hops, we see a close correlation. Figure 8 and 9 (see appendix) are the loss and accuracy charts that we got from 20 task.

Some select comparison with the original paper is tasks numbered 1,6,16,18, and 19. In task 1, single supporting fact, the paper claimed that they got 99% accuracy with 3 hops, while we got 64% with the same structure. One thing to note here is that we only have the capacity to use bag-of-words, while the paper looks at the problem from different angles. This is why the chart that's provided by the paper has many columns. In task 6, counting, we get 75% accuracy, and the paper got 82%. In task 16, basic induction, we have 45% accuracy, while the paper gets similar results with 1,2 hops, but an enormously good result of 96% with 3 hops. In task 18, size reasoning, we get a really good result of 92%, whereas the paper got 91%. The task number 19, path finding, is the worst performed model in both of our implementations, which is no more than 10%

Finally, the paper also comments on how the models behave when they use all the tasks jointly. If we look at our results, we again, see a good correlation in the accuracy matrix.

### Authors
Syed Muhammed Hasan Rizvi hasan@hawk.iit.edu

Ekrem Guzelyel eguzelyel@hawk.iit.edu


### References
Hui, J. Memory Network. J​onathan hui blog.​Retrieved from jhui.github.io/2017/03/15/Memory-network/

Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin and Tomas Mikolov. ​Towards AI Complete Question Answering: A Set of Prerequisite Toy Tasks,​ arXiv​:1502.05698.

Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. 2015. End-to-end memory networks. In Advances in neural information processing systems, pages 2440–2448.

Weng, L. Attention? Attention! L​il-log​. Retrieved from lilianweng.github.io/lil-log​/2018/06/24/a​ ttention-attention.html

Weston, Jason, Chopra, Sumit, and Bordes, Antoine. Memory networks. CoRR​, abs/1410.3916, 2014.

