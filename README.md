# AutoMaster-Dialogue2Report-AutoAbstract

Through the training on dataset of Automobile's dialogue between customers and the service can we 
generate a report automatically.  
Overall structure uses Seq2Seq with attention, and embedded
with pre-trained word vector of 100 dim.   
The translation uses two solutions of beam search and greedy search.
 
  
some experience:   
1. Size.  
small-sized vocab(3000) with 100dim word vector, 10 
encoding units reduce the loss to around 1.1 in 10 epoches
, which is very quick. But the result didn't shows good, it can hardly recognized as a semantic 
sentence. After trying, a vocab of 20k and 256dim word vector, 200 encoding
units had a good performance, it is my baseline for this project.
2. Pretrained word vector.  
With pretrained word2vec didn't result in a faster reduction 
on loss, on the other hand, with trainable embedding layer result in
a faster reduction  
3. Batch-size  
At a 20k vocab size, 256 word vector dim and 200 hidden units, with
8 batch-size (concerning GPU memory limit) it nearly stopped reduction on loss after 30 epoches(or say a o(10^-3) 
speed) at around 1.25 and training time for whole dataset costs 1300s. While it had a bounce when increasing to 
16 batch-size, loss were about 1.1, and half reduction on training time
to 600s, increasing to 32 batch-size caused a little jump as well but not
much, while training time stayed at 480s, which wasn't another half
of 600.  
It shows that if GPU memory supports, bigger batch-size can help the 
model learn better on data, with faster speed.  
4. Beam search and greedy search  
The method is not so much better than greedy search, probably 
because the training process has always been like a greedy search
and it trains pretty well on the ability of deduction on one-to-one word
way, instead of a consideration on global optimization of beam search.