import os
# from sklearn.cluster import KMeans
# from rouge import Rouge
import pandas as pd
import random
import numpy as np


# from sentence_transformers import SentenceTransformer
# def get_knowledge_set(raw_texts, r = 0.1):
#
#     rouge = Rouge()
#     total_num_sample = len(raw_texts)
#     num_selected = int(total_num_sample * r)
#     indexes = list(range(total_num_sample))
#     random.shuffle(indexes)
#     ref_index_list = []
#     cnt = 0
#     thred =0.7
#     for index in indexes:
#         if cnt == 0:
#             ref_index_list.append(index)
#             cnt+=1
#         else:
#             max_rouge_l = 0
#             for i in range(cnt):
#                 a = raw_texts[index]
#                 b = raw_texts[ref_index_list[i]]
#                 score = rouge.get_scores(a, b)[0]["rouge-l"]['f']
#                 max_rouge_l = max(max_rouge_l, score)
#             if max_rouge_l < thred:
#                 ref_index_list.append(index)
#                 cnt+=1
#         if cnt >= num_selected:
#             break
#     return ref_index_list
#
# def get_cluster_centers(data, k = 5):
#
#     kmeans = KMeans(n_clusters = k)
#     kmeans.fit(data)
#     labels = kmeans.labels_ #输出每一样本的聚类的类簇标签
#     centers = kmeans.cluster_centers_ #输出聚类的类簇中心点
#     return centers, labels

class InstructionsHandler:
    def __init__(self, config=None):
        self.ate = {}
        self.ote = {}
        self.atsc = {}
        self.aspe = {}
        self.aooe = {}
        self.aope={}
        self.aoste = {}
        self.aos={}
    #     self.aoste_def = "Definition: The output will be the aspects (explicit) the corresponding opinion terms (explicit) and the sentiment polarity (positive, negative, neutral) of the opinion term . In cases where there are no aspects the output should be noaspectterm:none:none."
    #     self.knowledge_set = {}
    #     self.knowledge_embeddings = {}
    #     self.knowledge_clusters = {}
    #     self.num_cluster = config.k
    #     self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    #     if os.path.exists(config.knowledge_path+'/res14_knowledge.csv'):
    #         self.knowledge_set['res14'],self.knowledge_embeddings['res14'],self.knowledge_clusters['res14'] = \
    #             self.load_knowledge_set(config.knowledge_path, config.knowledge_path+'/res14_knowledge.csv', name_pre='res14')
    #     else:
    #         if not os.path.exists(config.knowledge_path):
    #             os.makedirs(config.knowledge_path)
    #         self.knowledge_set['res14'],self.knowledge_embeddings['res14'],self.knowledge_clusters['res14'] = \
    #             self.create_knowledge_set(config.knowledge_path, config.knowledge_res_path, r = config.r, k = config.k, name_pre='res14')
    #
    #     if os.path.exists(config.knowledge_path+'/lap14_knowledge.csv'):
    #         self.knowledge_set['lap14'],self.knowledge_embeddings['lap14'],self.knowledge_clusters['lap14'] = \
    #             self.load_knowledge_set(config.knowledge_path, config.knowledge_path+'/lap14_knowledge.csv', name_pre='lap14')
    #     else:
    #         if not os.path.exists(config.knowledge_path):
    #             os.makedirs(config.knowledge_path)
    #         self.knowledge_set['lap14'],self.knowledge_embeddings['lap14'],self.knowledge_clusters['lap14'] = \
    #             self.create_knowledge_set(config.knowledge_path, config.knowledge_lap_path, r=config.r, k=config.k, name_pre='lap14')
    # def create_knowledge_set(self, knowledge_path, id_tr_data_path, r=0.1, k=6, name_pre = 'lap14'):
    #     id_tr_df = pd.read_csv(id_tr_data_path)
    #     ref_index_list = get_knowledge_set(list(id_tr_df['raw_text']), r=r)
    #     knowledge_set = id_tr_df.iloc[ref_index_list]
    #     input_output = []
    #     knowledge_embeddings = self.sentence_encoder.encode(list(knowledge_set['raw_text']))
    #     centers, labels = get_cluster_centers(knowledge_embeddings, k=k)
    #     knowledge_set = knowledge_set.assign(cluster_id=labels)
    #     knowledge_set.to_csv(knowledge_path+'/'+name_pre+"_knowledege.csv", index=False)
    #     np.savetxt(knowledge_path+'/' +name_pre+'_'+"cluster_centers.txt", centers, fmt='%f', delimiter=',')
    #
    #     return knowledge_set,knowledge_embeddings,centers
    #
    # def load_knowledge_set(self, knowledge_path, id_tr_data_path, name_pre='lap14'):
    #     knowledge_set = pd.read_csv(id_tr_data_path)
    #     knowledge_embeddings = self.sentence_encoder.encode(list(knowledge_set['raw_text']))
    #     centers = np.loadtxt(knowledge_path+'/' +name_pre+'_'+"cluster_centers.txt", delimiter=',')
    #     return knowledge_set,knowledge_embeddings,centers
    def load_instruction_set1(self, ):
        ################################# AOS #################################
        self.aos['bos_instruct1'] = """The output will be 'positive' if the aspect-opinion pair identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect-opinion pair in the input is negative the answer will be 'negative'. 
                Otherwise, the output should be 'neutral'. 
                        Positive example 1-
                        input: I charge it at night and skip taking the cord with me because of the good battery life. The aspect-opinion pair is: (battery life, good).
                        output: positive
                        Positive example 2-
                        input: I just wonder how you can have such a delicious meal for such little money . The aspect-opinion pair is: (money, little).
                        output: positive
                        Now complete the following example-
                        input: """

        self.aos['delim_instruct'] = ''
        self.aos['eos_instruct'] = '. \noutput:'
        ################################# ATE #################################
        self.ate['definition'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm."""
        self.ate['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text.
                Positive example 1-
                input: I charge it at night and skip taking the cord with me because of the good battery life.
                output: battery life
                Positive example 2-
                input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                output: features, iChat, Photobooth, garage band
                Now complete the following example-
                input: """

        self.ate['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) which have associated opinions that are extracted from the input text.
                Positive example 1-
                input: With the great variety on the menu , I eat here often and never get bored.
                output: menu
                Positive example 2- 
                input: Great food, good size menu, great service and an unpretensious setting.
                output: food, menu, service, setting
                Now complete the following example-
                input: """
        self.ate['delim_instruct'] = ''
        self.ate['eos_instruct'] = ' \noutput:'
        ################################# OTE #################################

        self.ote['bos_instruct1'] = """Definition: The output will be the opinions (both implicit and explicit) which have associated aspects that are extracted from the input text.
                        Positive example 1-
                        input: I charge it at night and skip taking the cord with me because of the good battery life.
                        output: good
                        Positive example 2-
                        input: Great food, good size menu, great service and an unpretensious setting.
                        output: Great, good size, great, unpretensious
                        Now complete the following example-
                        input: """

        self.ote['delim_instruct'] = ''
        self.ote['eos_instruct'] = ' \noutput:'
        ################################# ATSC #################################
        self.atsc['definition'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                        Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none."""
        self.atsc['bos_instruct1'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
                Positive example 1-
                input: I charge it at night and skip taking the cord with me because of the good battery life. The aspect is battery life.
                output: positive
                Positive example 2-
                input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!. The aspect is garage band.
                output: positive
                Now complete the following example-
                input: """

        self.atsc['bos_instruct2'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
                Positive example 1-
                input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
                output: positive
                Positive example 2- 
                input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
                output: positive
                Now complete the following example-
                input: """
        self.atsc['delim_instruct'] = ' The aspect is '
        self.atsc['eos_instruct'] = '.\noutput:'

        ################################# ASPE #################################


        self.aspe['definition'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none."""
        self.aspe['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity.
                Positive example 1-
                input: I charge it at night and skip taking the cord with me because of the good battery life.
                output: battery life:positive 
                explanation: battery life:good:positive
                Positive example 2-
                input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
                explanation: none
                Now complete the following example-
                input: """

        self.aspe['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity.
                Positive example 1-
                input: With the great variety on the menu , I eat here often and never get bored.
                output: menu:positive
                Positive example 2- 
                input: Great food, good size menu, great service and an unpretensious setting.
                output: food:positive, menu:positive, service:positive, setting:positive
                Now complete the following example-
                input: """
        self.aspe['delim_instruct'] = ''
        self.aspe['eos_instruct'] = ' \noutput:'

        ################################# AOOE #################################

        self.aooe['bos_instruct1'] = """Definition: The output will be the opinion/describing word of the given aspect terms. In cases where there are no aspects the output should be none.
                Positive example 1-
                input: I charge it at night and skip taking the cord with me because of the good battery life . The aspect is battery life.
                output: good
                Positive example 2-
                input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous. The aspect is GUI.
                output: killer
                Now complete the following example-
                input: """

        self.aooe['bos_instruct2'] = """Definition: The output will be the opinion/describing word for the aspect term in the sentence. In cases where there are no aspects the output should be none.
                Positive example 1-
                input: Faan 's got a great concept but a little rough on the delivery . The aspect term is delivery.
                output: rough
                Positive example 2- 
                input: At the end you 're left with a mild broth with noodles that you can slurp out of a cup . The aspect term is broth with noodles.
                output: mild
                Now complete the following example-
                input: """
        self.aooe['delim_instruct'] = ' The aspect is '
        self.aooe['eos_instruct'] = '.\noutput:'

        ################################# AOPE #################################

        self.aope['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) and the corresponding opinion/describing terms. In cases where there are no aspects the output should be noaspectterm:none.
                Positive example 1-
                input: I charge it at night and skip taking the cord with me because of the good battery life.
                output: battery life:good 
                Positive example 2-
                input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous.
                output: quality:high, GUI:killer, applications:good, use:easy 
                Now complete the following example-
                input: """

        self.aope['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
                Positive example 1-
                input: Faan 's got a great concept but a little rough on the delivery .
                output: delivery:rough
                Positive example 2- 
                input: I just wonder how you can have such a delicious meal for such little money .
                output: meal:delicious, money:little
                Now complete the following example-
                input: """
        self.aope['delim_instruct'] = ''
        self.aope['eos_instruct'] = ' \noutput:'

        ################################# AOSTE #################################
        definition = "Definition: The output will be the aspects (explicit) the corresponding opinion terms (explicit) and the sentiment polarity (positive, negative, neutral) of the opinion term  in the given text. The format of the output will follow the order of aspect term: opinion term: sentiment polarity. "
        example = """Positive example 1-
            input: lots of extra space but the keyboard is ridiculously small .
            output: space:lots:positive, keyboard:small:negative
            Positive example 2-
            input: I do not experience a lot of heat coming out of it , however I would highly suggest purchasing a stand however , due to the nature of the design of the macbook as it is one very large heat sink .
            output: stand:suggest:neutral, heat sink:large:negative
            """
        self.aoste['definition'] = definition
        self.aoste['bos_instruct1'] = definition + '\n' + example + "\nNow complete the following example-" + "\ninput: "


        self.aoste['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) the corresponding opinion/describing terms and the sentiment polarity (positive, negative, neutral) of the opinion term . In cases where there are no aspects the output should be noaspectterm:none:none.
        Positive example 1-
        input: Faan 's got a great concept but a little rough on the delivery .
        output: delivery:rough:positive
        Positive example 2-
        input: I just wonder how you can have such a delicious meal for such little money .
        output: meal:delicious:positive, money:little:positive
        Now complete the following example-
        input: """
        self.aoste['delim_instruct'] = ''
        self.aoste['eos_instruct'] = '\noutput: '

    def load_instruction_set2(self, ):

        ################################# ATE #################################

        self.ate['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features, iChat, Photobooth, garage band
        Negative example 1-
        input: Speaking of the browser, it too has problems.
        output: browser
        Negative example 2-
        input: The keyboard is too slick.
        output: keyboard
        Neutral example 1-
        input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
        output: battery
        Neutral example 2-
        input: Nightly my computer defrags itself and runs a virus scan.
        output: virus scan
        Now complete the following example-
        input: """

        self.ate['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food, menu, service, setting
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
        output: toast, mayonnaise, bacon, ingredients, plate
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
        output: seats
        Neutral example 1-
        input: I asked for seltzer with lime, no ice.
        output: seltzer with lime
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another.
        output: glass of wine
        Now complete the following example-
        input: """
        self.ate['delim_instruct'] = ''
        self.ate['eos_instruct'] = ' \noutput:'

        ################################# ATSC #################################

        self.atsc['bos_instruct1'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
        Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life. The aspect is battery life.
        output: positive
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!. The aspect is garage band.
        output: positive
        Negative example 1-
        input: Speaking of the browser, it too has problems. The aspect is browser.
        output: negative
        Negative example 2-
        input: The keyboard is too slick. The aspect is keyboard.
        output: negative
        Neutral example 1-
        input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset. The aspect is battery.
        output: neutral
        Neutral example 2-
        input: Nightly my computer defrags itself and runs a virus scan. The aspect is virus scan.
        output: neutral
        Now complete the following example-
        input: """

        self.atsc['bos_instruct2'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
        Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
        output: positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
        output: positive
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it. The aspect is toast.
        output: negative
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches. The aspect is seats.
        output: negative
        Neutral example 1-
        input: I asked for seltzer with lime, no ice. The aspect is seltzer with lime.
        output: neutral
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another. The aspect is glass of wine.
        output: neutral
        Now complete the following example-
        input: """
        self.atsc['delim_instruct'] = ' The aspect is '
        self.atsc['eos_instruct'] = '.\noutput:'

        ################################# ASPE #################################

        self.aspe['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:positive, 
        Positive example 2-
        input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
        Negative example 1-
        input: Speaking of the browser, it too has problems.
        output: browser:negative
        Negative example 2-
        input: The keyboard is too slick.
        output: keyboard:negative
        Neutral example 1-
        input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
        output: battery:neutral
        Neutral example 2-
        input: Nightly my computer defrags itself and runs a virus scan.
        output: virus scan:neutral
        Now complete the following example-
        input: """

        self.aspe['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: With the great variety on the menu , I eat here often and never get bored.
        output: menu:positive
        Positive example 2- 
        input: Great food, good size menu, great service and an unpretensious setting.
        output: food:positive, menu:positive, service:positive, setting:positive
        Negative example 1-
        input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
        output: toast:negative, mayonnaise:negative, bacon:negative, ingredients:negative, plate:negative
        Negative example 2-
        input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
        output: seats:negative
        Neutral example 1-
        input: I asked for seltzer with lime, no ice.
        output: seltzer with lime:neutral
        Neutral example 2-
        input: They wouldnt even let me finish my glass of wine before offering another.
        output: glass of wine:neutral
        Now complete the following example-
        input: """
        self.aspe['delim_instruct'] = ''
        self.aspe['eos_instruct'] = ' \noutput:'

        ################################# AOOE #################################

        self.aooe['bos_instruct1'] = """Definition: The output will be the opinion/describing word of the aspect terms in the sentence. In cases where there are no aspects the output should be none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life . The aspect is battery life.
        output: good
        Positive example 2-
        input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous. The aspect is GUI.
        output: killer
        Negative example 1-
        input: One night I turned the freaking thing off after using it , the next day I turn it on , no GUI , screen all dark , power light steady , hard drive light steady and not flashing as it usually does . The aspect is GUI.
        output: no
        Negative example 2-
        input: I can barely use any usb devices because they will not stay connected properly . The aspect is usb devices.
        output: not stay connected properly
        Neutral example 1-
        input: However , the multi-touch gestures and large tracking area make having an external mouse unnecessary ( unless you 're gaming ) . The aspect is external mouse.
        output: unnecessary
        Neutral example 2-
        input: I wanted to purchase the extended warranty and they refused , because they knew it was trouble . The aspect is extended warranty.
        output: refused
        Now complete the following example-
        input: """

        self.aooe['bos_instruct2'] = """Definition: The output will be the opinion/describing word of the aspect terms in the sentence. In cases where there are no aspects the output should be none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life . The aspect is battery life.
        output: good
        Positive example 2-
        input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous. The aspect is GUI.
        output: killer
        Negative example 1-
        input: The menu is very limited - i think we counted 4 or 5 entrees . The aspect is menu.
        output: limited
        Negative example 2-
        input: The strong scents coming from the left and right of me negatively affected my taste buds . The aspect is scents.
        output: strong
        Neutral example 1-
        input: What came to our table was burned beyond recognition and stringy . The aspect is battery table.
        output: burned
        Neutral example 2-
        input: But , nothing stands out about the cooking . The aspect is cooking.
        output: nothing stands out
        Now complete the following example-
        input: """
        self.aooe['delim_instruct'] = ' The aspect is '
        self.aooe['eos_instruct'] = '.\noutput:'

        ################################# AOPE #################################

        self.aope['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) and the corresponding opinion/describing terms. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:good 
        Positive example 2-
        input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous.
        output: quality:high, GUI:killer, applications:good, use:easy
        Negative example 1-
        input: A month or so ago , the freaking motherboard just died .
        output: motherboard:freaking, motherboard:freaking
        Negative example 2-
        input: I had always used PCs and been constantly frustrated by the crashing and the poorly designed operating systems that were never very intuitive .
        output: operating systems:poorly designed, operating systems:intuitive
        Neutral example 1-
        input: It has a 10 hour battery life when you 're doing web browsing and word editing , making it perfect for the classroom or office , and in terms of gaming and movie playing it 'll have a battery life of just over 5 hours .
        output: web browsing:perfect, word editing:perfect
        Neutral example 2-
        input: no complaints with their desktop , and maybe because it just sits on your desktop , and you do n't carry it around , which could jar the hard drive , or the motherboard .
        output: hard drive:jar, motherboard:jar
        Now complete the following example-
        input: """

        self.aope['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
        Positive example 1-
        input: Faan 's got a great concept but a little rough on the delivery .
        output: delivery:rough
        Positive example 2- 
        input: I just wonder how you can have such a delicious meal for such little money .
        output: meal:delicious, money:little
        Negative example 1-
        input: From the terrible service , to the bland food , not to mention the unaccommodating managers , the overall experience was horrible .
        output: service:terrible, food:bland, managers:unaccommodating
        Negative example 2- 
        input: I had the Pad Thai and the noodles were sticky .
        output: Pad Thai:sticky, noodles:sticky
        Neutral example 1-
        input: The Dim Sum was so-so , but not spectacular .
        output: Dim Sum:so-so, Dim Sum:not spectacular
        Neutral example 2- 
        input: The location and ambience is Ok but the food is what makes up for it .
        output: mlocationeal:Ok, ambience:Ok
        Now complete the following example-
        input: """
        self.aope['delim_instruct'] = ''
        self.aope['eos_instruct'] = ' \noutput:'

        ################################# AOSTE #################################

        self.aoste['bos_instruct1'] = """Definition: The output will be the aspects (explicit) the corresponding opinion terms (explicit) and the sentiment polarity (positive, negative, neutral) of the opinion term  in the given text. The format of the output will follow the order of aspect term, opinion term, sentiment polarity. 
        Positive example 1-
        input: I charge it at night and skip taking the cord with me because of the good battery life.
        output: battery life:good:positive
        Positive example 2-
        input: it is of high quality , has a killer GUI , is extremely stable , is highly expandable , is bundled with lots of very good applications , is easy to use , and is absolutely gorgeous.
        output: quality:high:positive, GUI:killer:positive, applications:good:positive, use:easy:positive
        Negative example 1-
        input: A month or so ago , the freaking motherboard just died .
        output: motherboard:freaking:negative, motherboard:freaking:negative
        Negative example 2-
        input: I had always used PCs and been constantly frustrated by the crashing and the poorly designed operating systems that were never very intuitive .
        output: operating systems:poorly designed:negative, operating systems:intuitive:negative
        Neutral example 1-
        input: It has a 10 hour battery life when you 're doing web browsing and word editing , making it perfect for the classroom or office , and in terms of gaming and movie playing it 'll have a battery life of just over 5 hours .
        output: web browsing:perfect:neutral, word editing:perfect:neutral
        Neutral example 2-
        input: no complaints with their desktop , and maybe because it just sits on your desktop , and you do n't carry it around , which could jar the hard drive , or the motherboard .
        output: hard drive:jar:neutral, motherboard:jar:neutral
        Now complete the following example-
        input: """
        # self.aoste['bos_instruct1'] = """Definition: The output will be the aspects (explicit) the corresponding opinion terms (explicit) and the sentiment polarity (positive, negative, neutral) of the opinion term . In cases where there are no aspects the output should be noaspectterm:none:none.
        #                 Positive example 1-
        #                 input: Keyboard responds well to presses .
        #                 output: Keyboard: responds well: positive
        #                 Positive example 2-
        #                 input: Strong build though which really adds to its durability .
        #                 output: durability: Strong: positive,  build: Strong: positive
        #                 Negative example 1-
        #                 input: When I finally had everything running with all my software installed I plugged in my droid to recharge and the system crashed .
        #                 output: system:crashed:negative
        #                 Negative example 2-
        #                 input: The machine is slow to boot up and occasionally crashes completely .
        #                 output: boot up:slow:negative
        #                 Now complete the following example-
        #                 input: """
        self.aoste['bos_instruct2'] = """Definition: The output will be the aspects (explicit) the corresponding opinion terms (explicit) and the sentiment polarity (positive, negative, neutral) of the opinion term . In cases where there are no aspects the output should be noaspectterm:none:none.
        Positive example 1-
        input: Faan 's got a great concept but a little rough on the delivery .
        output: delivery:rough:positive
        Positive example 2- 
        input: I just wonder how you can have such a delicious meal for such little money .
        output: meal:delicious:positive, money:little:positive
        Negative example 1-
        input: From the terrible service , to the bland food , not to mention the unaccommodating managers , the overall experience was horrible .
        output: service:terrible:negative, food:bland:negative, managers:unaccommodating:negative
        Negative example 2- 
        input: I had the Pad Thai and the noodles were sticky .
        output: Pad Thai:sticky:negative, noodles:sticky:negative
        Neutral example 1-
        input: The Dim Sum was so-so , but not spectacular .
        output: Dim Sum:so-so:neutral, Dim Sum:not spectacular:neutral
        Neutral example 2- 
        input: The location and ambience is Ok but the food is what makes up for it .
        output: mlocationeal:Ok:neutral, ambience:Ok:neutral
        Now complete the following example-
        input: """
        self.aoste['delim_instruct'] = ''
        self.aoste['eos_instruct'] = ' \noutput:'