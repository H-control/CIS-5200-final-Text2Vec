**Related works**

Document segmentation plays a central role in modern retrieval pipelines, particularly in Retrieval-Augmented Generation (RAG) systems in which dense representations of text chunks are indexed for semantic search. Early approaches predominantly used fixed-size sliding windows, motivated by computational simplicity and compatibility with transformer token limits. However, prior studies demonstrate that fixed-size chunking may disrupt topical coherence, causing embeddings to encode incomplete or fragmented semantic units (Karpukhin et al., 2020). This fragmentation negatively affects retrieval precision because the model’s representation is constrained by arbitrary window boundaries rather than natural discourse structure.

To mitigate this, a growing line of work investigates structure-aware segmentation, especially for documents with intrinsic HTML or hierarchical organization. Research on Wikipedia-based QA datasets, including Natural Questions (Kwiatkowski et al., 2019) and MS MARCO (Bajaj et al., 2016), highlights that respecting HTML or section boundaries better aligns chunk boundaries with meaningful semantic divisions. Practical RAG systems often enforce paragraph-level or header-based segmentation for similar reasons. Although these methods improve semantic coherence, they face limitations: structural boundaries do not always correspond to topical boundaries, and sections may vary widely in length, leading to uneven chunk sizes and potential incompleteness.

Another direction is semantic-driven segmentation, where chunk boundaries are identified using similarities between adjacent sentences or embeddings. Methods based on sentence-level representations (Reimers & Gurevych, 2019) or topic shift detection attempt to maintain high intra-chunk semantic coherence while reducing unnecessary overlap. These methods are more adaptive than fixed windows and more context-sensitive than purely structural approaches. However, they are computationally expensive, require accurate similarity metrics, and can be sensitive to noise or formatting inconsistencies in HTML documents.

A more recent advancement is late chunking, introduced by Günther (2024). Instead of segmenting before embedding, late chunking first generates embeddings over long or full-context sequences and determines chunk boundaries in the embedding space. This allows each chunk to benefit from global document context while still enabling fine-grained retrieval units. Günther and Xiao (2024) further provide empirical evidence that late chunking yields better semantic separation and reduces context fragmentation compared to pre-chunking approaches, particularly in long-context embedding models. Despite these advantages, late chunking is computationally heavier and requires embedding models capable of handling long sequences.

Complementary work in dense retrieval and RAG architectures underscores the importance of chunk granularity. Studies on Dense Passage Retrieval (DPR) (Karpukhin et al., 2020), long-context transformers (Xiong et al., 2021), and hierarchical retrieval frameworks (Khattab & Zaharia, 2020) consistently show that retrieval quality depends heavily on the semantic coherence, length, and structure of the segmented units. Yet, there remains a lack of controlled, systematic comparison of chunking strategies on HTML-heavy corpora such as Natural Questions, especially under identical embedding, indexing, and retrieval conditions.

Taken together, existing literature exhibits a clear evolution: from fixed windows, to structure-aware chunking, to semantically guided segmentation, and finally to representation-level segmentation through late chunking. Each comes with trade-offs in semantic fidelity, computational efficiency, and robustness to structural heterogeneity. These gaps motivate the present study, which seeks to empirically evaluate the impact of chunking strategies on retrieval performance using long-context embedding models and a realistic HTML-based dataset.


Bajaj, P., Campos, D., Craswell, N., et al. (2016). MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.

Günther, M. (2024). Late Chunking: Chunk After Embedding. arXiv:2401.01352.

Günther, M., & Xiao, T. (2024). LateChunking Implementation Notes. arXiv preprint.

Karpukhin, V., Oguz, B., Min, S., et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550.pdf). EMNLP. 

Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR.

Kwiatkowski, T., Palomaki, J., Redfield, O., et al. (2019). [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026.pdf). TACL.

Reimers, N., & Gurevych, I. (2019). [Sentence-BERT: Sentence Embeddings Using Siamese BERT Networks](https://arxiv.org/pdf/1908.10084). EMNLP.

Xiong, W., et al. (2021). Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. ICLR.
