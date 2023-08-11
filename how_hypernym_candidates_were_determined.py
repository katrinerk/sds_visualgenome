# Katrin Erk June 2023
# hypernyms of objects in the Visual Genome


import sys
import os
import json
import zipfile
from collections import defaultdict, Counter
import math
import statistics
import random
import nltk
from nltk.corpus import wordnet

from vgnames import VGOBJECTS, VGATTRIBUTES, VGRELATIONS 
import vgiterator
import sentence_util
from vgpaths import VGPaths, get_output_path
from vec_util import VectorInterface
from sds_imagine_util import  ImagineAttr

print("reading data")
vgpath_obj = VGPaths()


# most frequent objects, attributes, relations
vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
with zipfile.ZipFile(vgcounts_zipfilename) as azip:
    with azip.open(vgcounts_filename) as f:
        vgobjects_attr_rel = json.load(f)

objectlabels = vgobjects_attr_rel[VGOBJECTS]

###
print("determining hypernyms")

hypernym_candidates = Counter()
hypfn = lambda s:s.hypernyms()

words_with_no_syn = [ ]
words_with_no_hyper = [ ]

for objectlabel in objectlabels:

    synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)
    # did we find any synsets?
    if len(synsets) == 0:
        objectlabel = objectlabel.split()[-1]
        synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)

        if len(synsets) == 0:
            words_with_no_syn.append(objectlabel)
            continue

    found = False
    for syn0 in synsets:

        # hypernyms
        hyper_synsets = list(syn0.closure(hypfn))

        # only retain hypernyms that contain a one-word lemma
        hyper_synsets_retained = [ ]
        for h in hyper_synsets:
            lemmas = [l.name() for l in h.lemmas()]
            lemmas = [l for l in lemmas if "_" not in l]
            if len(lemmas) > 0:
                hyper_synsets_retained.append(h)

        if len(hyper_synsets_retained) > 0:
            found = True
            break

    if not found:
        words_with_no_hyper.append(objectlabel)

print("objects with no synset:", len(words_with_no_syn))
print(words_with_no_syn)
print("\n\n")
print("objects with no single-word hypernyms:", len(words_with_no_hyper))
print(words_with_no_hyper)


# we get 95 words with no synset, many of them typos:
"""['advertisment', 'backsplash', 'badsentence', 'barcode', 'barefoot', 'big', 'biker', 'bookbag', 'bouy', 'bright', 'brocolli', 'buiding', 'buidling', 'building.', 'bulding', 'buliding', 'clocktower', 'cloudy', 'electronic', 'firetruck', 'flusher', 'foilage', "giraffe's", 'woma', 'graffitti', 'grafitti', 'grassy', 'her', 'homeplate', 'hoodie', 'hoove', 'indoors', 'iphone', 'is', 'kneepad', 'kneepads', 'lable', 'lightpost', 'long', 'macbook', "man's", 'motocycle', 'motorcyle', 'nightstand', 'on', 'parked', 'pavers', "person's", 'placemat', 'potatoe', 'powerline', 'powerlines', 'raquet', 'sandles', 'sandwhich', 'scissor', 'she', 'shelve', 'shelving', 'skatepark', 'skiier', 'skiiers', 'skiis', 'smartphone', 'snowpants', 'something', 'spacebar', 'stovetop', 'streetlamp', 'striped', 'surboard', 'tanktop', 'the', 'these', 'they', 'this', 'tiled', 'together', 'toliet', 'tomatoe', 'touchpad', 'trackpad', 'trashcan', 'tshirt', 'up', 'visible', 'wetsuit', 'whiteboard', 'wii', 'wiimote', 'wildebeast', 'windsheild', "woman's", 'wooden', 'writting']"""

# we only get 4 words with no hypernyms, which look rare enough:
"""['capris', 'googles', 'mit', 'nike']"""

###
# count how many hypernyms each word has,
# and how many object concepts each hypernym has

def all_hypernyms_of(objectlabel):

    # determine synsets
    synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)
    if len(synsets) == 0:
        objectlabel = objectlabel.split()[-1]
        synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)

        # no synsets found
        if len(synsets) == 0:
            return [ ]

    hyps = set()
    
    for syn0 in synsets:

        # hypernyms
        hyper_synsets = list(syn0.closure(hypfn))
        hyper_synsets_retained = set()
        for h in hyper_synsets:
            lemmas = [l.name() for l in h.lemmas()]
            lemmas = [l for l in lemmas if "_" not in l]
            if len(lemmas) > 0:
                hyper_synsets_retained.add(h)

        hyps.update(hyper_synsets_retained)

        
    return list(hyper_synsets_retained)

    
word_hyper = nltk.ConditionalFreqDist()
hyper_word = nltk.ConditionalFreqDist()


for objectlabel in objectlabels:
    hyps = all_hypernyms_of(objectlabel)

    for h in hyps:
        word_hyper[objectlabel][h] +=1
        hyper_word[h][objectlabel] += 1


# rare or overly common hypernyms, and words that only work with them
overlycommon = set([h for h in hyper_word.keys() if hyper_word[h].N() > 700])
rare = set([h for h in hyper_word.keys() if hyper_word[h].N() <= 10])
words_with_only_excluded_hyp = [w for w in word_hyper.keys() if all(h in overlycommon or h in rare for h in word_hyper[w].keys())]

for w in words_with_only_excluded_hyp:
    print(w, [(h, hyper_word[h].N()) for h in word_hyper[w].keys() if h in overlycommon])
print("Words that only have overly common or rare hypernyms", len(words_with_only_excluded_hyp))

##
# checking against a list of common words
# obtained from:
# https://github.com/first20hours/google-10000-english/blob/master/google-10000-english-usa.txt
# this is:
# "This repo contains a list of the 10,000 most common English words in order of frequency,
# as determined by n-gram frequency analysis of the Google's Trillion Word Corpus."

with open("google-10000-english-usa.txt") as f:
    frequentwords = [w.strip() for w in  f.readlines()]

# frequent hypernym words: don't include overly common words
fhyper = set()
for h in hyper_word.keys():
    if h in overlycommon or h in rare: continue
        
    lemmas = [l.name() for l in h.lemmas()]
    if any( l in frequentwords for l in lemmas):
        fhyper.add(h)

        
# how many words aren't covered by frequent-word hypernyms that have a medium number of instances?
nonf_obj = [w for w in word_hyper.keys() if all(h not in fhyper for h in word_hyper[w].keys())]

print(nonf_obj)
# for w in nonf_obj:
#     print(w, [(h, hyper_word[h].N()) for h in word_hyper[w].keys()])
print("Words that only have overly common or rare hypernyms, or hypernyms that aren't common words:", len(nonf_obj), "out of", len(objectlabels))

"""Words that only have overly common or rare hypernyms, or hypernyms that aren't common words: 144 out of 3514"""
"""['balloon', 'balloons', 'bar stool', 'barrier', 'barriers', 'baseboard', 'basin', 'bead', 'beads', 'binder', 'blanket', 'blankets', 'blue paint', 'braid', 'broccoli floret', 'bud', 'buds', 'button', 'buttons', 'carpet', 'ceramic', 'chandelier', 'cheese cube', 'clay', 'cleat', 'cleats', 'cloth', 'cloths', 'cobblestone', 'concrete', 'cone', 'cones', 'connector', 'container', 'containers', 'cork', 'cracker', 'crackers', 'cream', 'cube', 'curtain', 'curtains', 'cushion', 'cushions', 'decor', 'dolly', 'door knob', 'double decker', 'earth', 'equipment', 'flooring', 'floret', 'flower petal', 'foliage', 'fountain', 'frog', 'frond', 'frosting', 'furniture', 'garnish', 'goods', 'graffiti', 'greenery', 'group', 'growth', 'happy little paint', 'ice cream', 'jewelry', 'knob', 'knobs', 'lemon', 'lemons', 'lipstick', 'mat', 'mattress', 'moon', 'mousepad', 'orange cone', 'ornament', 'ornaments', 'padding', 'paint', 'parking space', 'pebble', 'pebbles', 'pencil', 'pencils', 'pendant', 'petal', 'petals', 'pig', 'pillow', 'pillows', 'place mat', 'plastic container', 'point', 'power button', 'railroad', 'railway', 'red paint', 'ribs', 'safety cone', 'satellite', 'scarf', 'shower curtain', 'siding', 'sill', 'slab', 'slat', 'slats', 'space', 'stool', 'stools', 'table cloth', 'tassel', 'telephone', 'tent', 'tents', 'thing', 'things', 'throw pillow', 'thumb', 'tick', 'tier', 'toiletries', 'topping', 'toppings', 'traffic cone', 'twig', 'twigs', 'unit', 'urinal', 'urinals', 'vehicle', 'vehicles', 'wedge', 'whipped cream', 'white button', 'white cloth', 'white frosting', 'white paint', 'white pillow', 'window sill', 'yellow paint']"""

#######3
# what is the list of all hypernyms that we are left with?
hypwords = set()
for h in hyper_word.keys():
    if h in fhyper:
        hypwords.add(h.name())

print("List of all hypernyms retained:", hypwords)

"""{'weapon.n.01', 'concept.n.01', 'collection.n.01', 'woman.n.01', 'shape.n.02', 'bodily_process.n.01', 'juvenile.n.01', 'device.n.01', 'traveler.n.01', 'mechanism.n.05', 'phenomenon.n.01', 'side.n.05', 'food.n.01', 'worker.n.01', 'boat.n.01', 'nutriment.n.01', 'seat.n.03', 'seat.n.04', 'knowledge_domain.n.01', 'vessel.n.03', 'opening.n.10', 'idea.n.01', 'sexual_activity.n.01', 'happening.n.01', 'produce.n.01', 'state.n.02', 'container.n.01', 'band.n.07', 'furniture.n.01', 'character.n.08', 'herb.n.01', 'message.n.02', 'substance.n.07', 'female.n.02', 'ware.n.01', 'process.n.06', 'solid.n.01', 'protective_covering.n.01', 'breath.n.01', 'housing.n.02', 'display.n.06', 'drug.n.01', 'craft.n.02', 'article.n.02', 'sport.n.01', 'award.n.02', 'canvas_tent.n.01', 'integer.n.01', 'group.n.01', 'restraint.n.06', 'fruit.n.01', 'case.n.05', 'liquid_body_substance.n.01', 'motion.n.06', 'vehicle.n.01', 'action.n.01', 'matter.n.03', 'equipment.n.01', 'product.n.02', 'region.n.01', 'creation.n.02', 'bird.n.01', 'writing.n.04', 'location.n.01', 'cake.n.03', 'magnitude.n.01', 'communication.n.02', 'machine.n.01', 'activity.n.01', 'organ.n.01', 'event.n.01', 'way.n.06', 'process.n.02', 'line.n.18', 'area.n.05', 'motion.n.03', 'movement.n.03', 'conveyance.n.03', 'relation.n.01', 'tool.n.01', 'thing.n.12', 'content.n.05', 'digit.n.01', 'part.n.02', 'boundary.n.01', 'structure.n.04', 'speech.n.02', 'strip.n.02', 'covering.n.01', 'organism.n.01', 'stroke.n.12', 'color.n.01', 'plant.n.02', 'fluid.n.01', 'cord.n.01', 'projection.n.04', 'car.n.01', 'position.n.09', 'jewelry.n.01', 'geological_formation.n.01', 'implement.n.01', 'animal.n.01', 'blow.n.01', 'cognition.n.01', 'structure.n.01', 'number.n.02', 'room.n.01', 'adult.n.01', 'substance.n.01', 'process.n.05', 'covering.n.02', 'design.n.04', 'barrier.n.01', 'cannabis.n.02', 'letter.n.02', 'measure.n.02', 'writing.n.02', 'diversion.n.01', 'mistake.n.01', 'gathering.n.01', 'discipline.n.01', 'travel.n.01', 'role.n.04', 'card.n.01', 'locomotion.n.02', 'appliance.n.02', 'attribute.n.02', 'change.n.03', 'shelter.n.01', 'hair.n.01', 'passage.n.03', 'fare.n.04', 'meat.n.01', 'causal_agent.n.01', 'material.n.01', 'surface.n.01', 'facility.n.01', 'support.n.10', 'food.n.02', 'vegetable.n.01', 'commodity.n.01', 'agent.n.03', 'instrument.n.01', 'inhalation.n.01', 'signal.n.01', 'tract.n.01', 'component.n.03', 'part.n.01', 'shell.n.08', 'building.n.01', 'region.n.03', 'part.n.03', 'representation.n.02', 'quality.n.01', 'person.n.01', 'system.n.01', 'paper.n.01', 'tent.n.01', 'act.n.02', 'time_period.n.01', 'sheet.n.06', 'symbol.n.01', 'property.n.02', 'mound.n.04', 'clothing.n.01', 'end.n.01', 'work.n.01', 'liquid.n.01', 'piece.n.01', 'aid.n.02', 'vessel.n.02', 'fabric.n.01'}"""



#####################3

# trainperc = 0.8

# print("reading data")
# # vgpath_obj = VGPaths(vgdata = args.vgdata)
# vgpath_obj = VGPaths()

# # most frequent objects, attributes, relations
# vgcounts_zipfilename, vgcounts_filename = vgpath_obj.vg_counts_zip_and_filename()
# with zipfile.ZipFile(vgcounts_zipfilename) as azip:
#     with azip.open(vgcounts_filename) as f:
#         vgobjects_attr_rel = json.load(f)

# vec_obj = VectorInterface(vgpath_obj)

# available_objects = [o for o in vgobjects_attr_rel[VGOBJECTS] if o in vec_obj.object_vec]
# missing = len(vgobjects_attr_rel[VGOBJECTS]) - len(available_objects)
# if missing > 0:
#     print("frequent objects without vectors:", missing, "out of", len(vgobjects_attr_rel[VGOBJECTS]))

# # train/dev/test split
# print("splitting objects into train/dev/test")
# random.seed(9386)
# training_objectlabels = random.sample(available_objects, int(trainperc * len(available_objects)))
# nontraining_objectlabels = [ o for o in available_objects if o not in training_objectlabels]
# dev_objectlabels = random.sample(nontraining_objectlabels, int(0.5 * len(nontraining_objectlabels)))
# test_objectlabels = [o for o in nontraining_objectlabels if o not in dev_objectlabels]

# ###
# print("determining hypernyms")

# hypernym_candidates = Counter()
# hypfn = lambda s:s.hypernyms()

# for objectlabel in training_objectlabels:

#     synsets = wordnet.synsets(objectlabel, pos = wordnet.NOUN)
#     # did we find any synsets?
#     if len(synsets) == 0:
#         continue

#     # first synset
#     syn0 = synsets[0]

#     # hypernyms
#     hyper_synsets = syn0.closure(hypfn)

#     # only retain hypernyms that contain a one-word lemma
#     hyper_synsets_retained = [ ]
#     for h in hyper_synsets:
#         lemmas = [l.name() for l in h.lemmas()]
#         lemmas = [l for l in lemmas if "_" not in l]
#         if len(lemmas) > 0:
#             hyper_synsets_retained.append(h)
    
#     # count these hypernyms
#     for h in hyper_synsets_retained:
#         hypernym_candidates[ h.name() ] += 1

# numhyp = 0
# for hyper, count in hypernym_candidates.most_common():
#     if count >= 140 or count < 10:
#         continue
#     numhyp += 1
#     print(hyper, wordnet.synset(hyper).lemmas(), count)

# print(numhyp)
