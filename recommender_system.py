import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import hebrew_tokenizer as ht
from wordcloud import WordCloud
from bidi.algorithm import get_display  # pip install python-bidi

ISRAELI_DATA_PATH = "csvs/israeli_data.csv"
MACRO_NUTRIENTS = ['protein', 'total_fat', 'carbohydrates']
MICRO_MINERALS = ['calcium', 'iron', 'magnesium', 'phosphorus', 'potassium', 'sodium', 'zinc', 'copper']
MICRO_VITAMINS = ['vitamin_a_iu', 'vitamin_e', 'vitamin_c', 'thiamin', 'riboflavin', 'niacin', 'vitamin_b6',
                  'folate', 'folate_dfe', 'vitamin_b12', 'carotene']
MICRO_NUTRIENTS = MICRO_MINERALS + MICRO_VITAMINS


def preprocess_data():
    data = pd.read_csv(ISRAELI_DATA_PATH)
    data = data[['smlmitzrach', 'shmmitzrach'] + MACRO_NUTRIENTS + MICRO_NUTRIENTS]
    data = data.fillna(0)  # to be replaced with the micronutrients predictions
    return data

def tokenize(hebrew_text):
    tokens = ht.tokenize(hebrew_text)  # tokenize returns a generator!
    return [
        token
        for grp, token, token_num, (start_index, end_index) in tokens
        if grp == 'HEBREW'
    ]


def get_recommendations(data, food_item, cosine_sim, indices):
    # Get the index of the food item
    idx = indices[food_item]

    # Get the pairwsie similarity scores of all foods with that food item
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the food items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 30 most similar food items
    sim_scores = sim_scores[1:30]

    # Get the food items indices
    foods_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar food items
    return data['shmmitzrach'].iloc[foods_indices]


# def recommneder_wordcloud(food_item, recommendations):
#     text = food_item
#     for recommended in recommendations:
#         text += " " + recommended
#     bidi_text = get_display(text)
#     wordcloud = WordCloud(font_path='C:\Windows\Fonts\courbd.ttf').generate(bidi_text)
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()


def find_food_item(data):
    food_item = input("Please enter a food item: ")  # Example: חלב 3% שומן, תנובה, טרה, הרדוף, יטבתה
    all_food_names = data['shmmitzrach'].squeeze()
    food_found = False
    if food_item in all_food_names.values:
        print("found food item")
        food_found = True
    else:
        print("did not find food item... looking for closest match")
        for name in all_food_names.values:
            if food_item in name:
                food_item = name
                food_found = True
                break
    if not food_found:
        exit("could not find a match for the given food item... exiting")
    return food_item


def content_base_recommender():
    data = preprocess_data()
    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfidf_matrix = tfidf.fit_transform(data['shmmitzrach'])
    cosine_sim_names = linear_kernel(tfidf_matrix, tfidf_matrix) # items names
    cosine_sim_macros = cosine_similarity(data[MACRO_NUTRIENTS])
    cosine_sim_nutri = cosine_similarity(data[MACRO_NUTRIENTS+MICRO_NUTRIENTS])
    indices = pd.Series(data.index, index=data['shmmitzrach']).drop_duplicates()
    food_item = find_food_item(data)

    print("----------first recommendations---------------")
    first_recommendation = get_recommendations(data, food_item, cosine_sim_names, indices)
    print(first_recommendation)
    # recommneder_wordcloud(food_item, first_recommendation)

    print("---------second recommendations----------")
    second_recommendation = get_recommendations(data, food_item, cosine_sim_macros, indices)
    print(second_recommendation)
    # recommneder_wordcloud(food_item, second_recommendation)

    print("---------third recommendations----------")
    third_recommendation = get_recommendations(data, food_item, cosine_sim_nutri, indices)
    print(third_recommendation)
    # recommneder_wordcloud(food_item, third_recommendation)


if __name__ == '__main__':
    content_base_recommender()
