# AutoMaster-Dialogue2Report-AutoAbstract

Through the training on dataset of Automobile's dialogue between customers and the service can we 
generate a report automatically.  
Overall structure uses Seq2Seq with attention, and embedded
with pre-trained word vector of 100 dim.   
The translation uses two solutions of beam search and greedy search.
  ___
   some example results:  
   * Q1:我的帕萨特烧机油怎么办怎么办？,技师说：你好，请问你的车跑了多少公里了，如果在保修期内，可以到当地的4店里面进行检查维修。如果已经超出了保修期建议你到当地的大型维修店进行检查，烧机油一般是发动机活塞环间隙过大和气门油封老化引起的。如果每7500公里烧一升机油的话，可以在后备箱备一些机油，以便机油报警时有机油及时补充，如果超过两升或者两升以上，建议你进行发动机检查维修。|技师说：你好|车主说：嗯,  
     
   * Re:你好 根据 你 的 描述 该车 发动机 烧 机油 如果 超出 了 建议 你 到 4S店 进行 检查 维修  
       
         

   * Q2:请问 那天右后胎扎了订，补了胎后跑高速80多开始有点抖，110时速以上抖动明显，以为是未做动平衡导致，做了一样抖，请问是不是前面两条胎的问题导致？,技师说：你好师傅！可能前轮平衡快脱落或者不平衡造成的！建议前轮做一下动平衡就好了！希望能够帮到你！祝你用车愉快！|车主说：谢谢大师！|技师说：不客气！祝您用车愉快！  
     
   * Re:你好 根据 你 说 的 这种 情况 建议 你 做 一下 轮胎 动平衡 就 可以 了
  ___  
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