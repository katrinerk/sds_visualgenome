import os

class VGPaths:
    # store general paths to use
    def __init__(self, vgpath = "/Users/kee252/Data/visual genome/",
                vgdata = "data/",
                sdsdata = "sds_in/vanilla/",
                sdsout = "sds_out/vanilla/"):
        self.visualgenome_path = vgpath
        self.datapath = vgdata
        self.sdsdatapath = sdsdata
        self.sdsoutpath = sdsout
        self.vg_manipulated_path = "changedvg/"

        self.vg_subpath_obj = "v1_2/"
        self.vg_subpath_attr = "v1_2/"
        self.vg_subpath_rel = "v1_2/"
        self.vgobjectfilename = "objects_v1_2.json.zip"
        self.vgattribfilename = "attributes.json.zip"
        self.vgrelfilename = "relationships_v1_2.json.zip"

    # location of visual genome objects file
    def vg_objects_zip_and_filename(self):
        return (os.path.join(self.visualgenome_path, self.vg_subpath_obj, self.vgobjectfilename),
                "objects.json")

    # location of visual genome attributes file
    def vg_attributes_zip_and_filename(self):
        return (os.path.join(self.visualgenome_path, self.vg_subpath_attr, self.vgattribfilename),
                "attributes.json")

    # location of visual genome relations file
    def vg_relations_zip_and_filename(self):
        return (os.path.join(self.visualgenome_path,  self.vg_subpath_rel, self.vgrelfilename),
                "relationships.json")
    

    # where to store identities of sufficiently frequent objects, attributes, relations
    def vg_counts_zip_and_filename(self):
        return (os.path.join(self.datapath, "vgcounts.json.zip"), "vgcounts.json")

    # where to store IDs of images in the training and test part
    def vg_traintest_zip_and_filename(self):
        return (os.path.join(self.datapath, "traintest.json.zip"), "traintest.json")

    # where to find VG-based vectors
    def vg_vecfilename(self):
        return os.path.join(self.datapath, "ext2vec.dm")

    #######3
    # gensim related paths
    # where to store the gensim dictionary
    def gensim_dict_zip_and_filename(self):
        return (os.path.join(self.datapath, "gensim_dict.pkl.zip"), "gensim_dict.pkl")

    # where to store the gensim corpus
    def gensim_corpus_zipfilename(self):
        return os.path.join(self.datapath, "gensim_corpus.zip")

    # names of individual files in the corpus for gensim
    def gensim_corpus_filename(self, prefix):
        return str(prefix) + ".json"

    # gensim output
    def gensim_out_zip_and_filenames(self):
        return (os.path.join(self.datapath, "gensim_output.zip"), "overall.json", "topics.json", "words.json", "topicword.json")

    ###########
    # files that will form input to SDS and output from SDS

    # zip files and inner files for parameter files for SDS
    def sds_filenames(self):
        return {"general" : os.path.join(self.sdsdatapath, "general.json"),
                "selpref" : (os.path.join(self.sdsdatapath, "selpref.json.gzip"), "selpref.json"),
                "scenario_concept" : (os.path.join(self.sdsdatapath, "scenario_concept.json.gzip"), "scenario_concept.json"),
                "word_concept" : (os.path.join(self.sdsdatapath, "word_concept.json.gzip"), "word_concept.json")
                }

    # zip file for sentences as input to SDS
    def sds_sentence_zipfilename(self):
        return (os.path.join(self.sdsdatapath, "sentences.zip"), "sentences.json")

    # zip file for where to write SDS output
    def sds_output_zipfilename(self):
        return (os.path.join(self.sdsoutpath, "sdsout.zip"), "results.json")


    # zip file and inner file for gold information for tasks
    def sds_gold(self):
        return (os.path.join(self.sdsdatapath, "gold.json.gzip"), "gold.json")
    
    ########
    # where to put manipulated object, attribute, relation files
    # as input to vector creation
    def vg_manipulated_filenames(self):
        return (os.path.join(self.vg_manipulated_path, self.vgobjectfilename),
                os.path.join(self.vg_manipulated_path, self.vgattribfilename),
                os.path.join(self.vg_manipulated_path, self.vgrelfilename))
