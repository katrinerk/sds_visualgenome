# sds_visualgenome
**Situation Description Systems applied to Visual Genome data**

Dependencies:
pgmax: need to install under Python 3.8
gensim

Needs cleanup:
Currently all paths are hardcoded in vgpaths.py. This will get cleaned up at some point. For now, edit it to set where you have the Visual Genome. 

*Preprocessing steps*:

* vgcounts: Determine sufficiently frequent objects, attributes, relations in the Visual Genome
* train_test_split: Split Visual Genome images into training and test portion
* make_topicmodeling_input, gensim_topic_modeling: Create scenarios as LDA topic models over VG images
* chane_vgdata_for_vecs: make version of the Visual Genome that only contains images in the training part of the split, retaining only sufficiently frequent objects, attributes, relations
* make_sds_input_cloze: Make input for SDS system, for the Cloze disambiguation task

*Situation Description Systems*:
* sds: create factor graphs, run MAP inference on given collection of sentences
* zoom_in: interactively inspect sentences, make factor graphs, run MAP inference, marginal inference and what-if analyses
