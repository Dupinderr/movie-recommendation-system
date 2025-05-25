#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

# Create a dataset of 50 movies
data = {
    'title': [
        'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump',
        'Inception', 'Fight Club', 'The Matrix', 'Interstellar', 'Gladiator',
        'Titanic', 'The Lord of the Rings: The Fellowship of the Ring', 'The Silence of the Lambs',
        'Se7en', 'The Green Mile', 'Saving Private Ryan', 'Schindler\'s List', 'Braveheart', 'The Lion King',
        'Avengers: Endgame', 'Joker', 'Toy Story', 'The Social Network', 'Deadpool', 'Black Panther',
        'Iron Man', 'Doctor Strange', 'Coco', 'Finding Nemo', 'Inside Out', 'Up', 'Frozen', 'The Incredibles',
        'Zootopia', 'Moana', 'Ratatouille', 'The Avengers', 'Guardians of the Galaxy', 'Spider-Man: Homecoming',
        'Logan', 'Wonder Woman', 'The Prestige', 'Django Unchained', 'Whiplash', 'La La Land', 'Parasite',
        '1917', 'The Revenant', 'Tenet'
    ],
    'genres': [
        'Drama', 'Crime|Drama', 'Action|Crime|Drama', 'Crime|Drama', 'Drama|Romance',
        'Action|Adventure|Sci-Fi', 'Drama', 'Action|Sci-Fi', 'Adventure|Drama|Sci-Fi', 'Action|Adventure|Drama',
        'Drama|Romance', 'Action|Adventure|Drama', 'Crime|Drama|Thriller', 'Crime|Drama|Mystery',
        'Crime|Drama|Fantasy', 'Drama|War', 'Biography|Drama|History', 'Biography|Drama|History',
        'Animation|Adventure|Drama', 'Action|Adventure|Drama', 'Crime|Drama|Thriller', 'Animation|Adventure|Comedy',
        'Biography|Drama', 'Action|Comedy', 'Action|Adventure|Sci-Fi', 'Action|Adventure|Sci-Fi',
        'Action|Adventure|Fantasy', 'Animation|Adventure|Family', 'Animation|Adventure|Comedy',
        'Animation|Adventure|Comedy', 'Animation|Adventure|Comedy', 'Animation|Adventure|Comedy',
        'Animation|Action|Adventure', 'Animation|Adventure|Comedy', 'Animation|Adventure|Comedy',
        'Animation|Comedy|Family', 'Action|Adventure|Sci-Fi', 'Action|Adventure|Comedy',
        'Action|Adventure|Sci-Fi', 'Action|Drama|Sci-Fi', 'Action|Adventure|Fantasy', 'Drama|Mystery|Sci-Fi',
        'Drama|Western', 'Drama|Music', 'Comedy|Drama|Music', 'Drama|Thriller', 'Drama|War',
        'Action|Adventure|Drama', 'Action|Sci-Fi'
    ],
    'description': [
        'Two imprisoned men bond and find redemption through acts of decency.',
        'An aging crime boss transfers control to his reluctant son.',
        'Batman faces the Joker, a mastermind plunging Gotham into chaos.',
        'Mobsters, a boxer, and a gangster\'s wife intersect in tales of crime.',
        'History through the eyes of a kind-hearted man with a low IQ.',
        'A thief steals corporate secrets through dream-sharing technology.',
        'An office worker and a soap maker start an underground fight club.',
        'A hacker discovers the true nature of reality and his role in it.',
        'Explorers travel through a wormhole to save humanity.',
        'A Roman general seeks vengeance against a corrupt emperor.',
        'A rich girl falls for a poor artist aboard the Titanic.',
        'A hobbit begins a journey to destroy a powerful ring.',
        'An FBI cadet consults a killer to catch another serial murderer.',
        'Two detectives hunt a killer using the seven deadly sins.',
        'Death row guards are changed by a gentle inmate\'s powers.',
        'Soldiers go behind enemy lines to save a paratrooper.',
        'A man saves Jews by employing them during the Holocaust.',
        'A Scottish warrior leads a rebellion against English rule.',
        'A lion prince must embrace his destiny to lead the kingdom.',
        'The Avengers try to reverse Thanos\' destruction.',
        'A comedian descends into madness and becomes the Joker.',
        'Toys deal with jealousy and identity when a new toy arrives.',
        'The rise of Facebook and the lawsuits that followed.',
        'A mercenary becomes Deadpool after a rogue experiment.',
        'T\'Challa returns to Wakanda to become king.',
        'An engineer builds a suit to escape captivity and become Iron Man.',
        'A surgeon learns mystical arts after a career-ending accident.',
        'A boy travels to the Land of the Dead to learn about family.',
        'A clownfish searches for his missing son across the ocean.',
        'A girl\'s emotions struggle with a life-changing move.',
        'An old man flies his house to South America with a boy scout.',
        'A queen must learn to control her icy powers.',
        'Superhero family returns to save the world again.',
        'A bunny cop and fox con artist uncover a conspiracy.',
        'A Polynesian girl sails to save her island.',
        'A rat becomes a chef in a top French restaurant.',
        'Earth\'s mightiest heroes unite to stop Loki\'s invasion.',
        'Criminals team up to stop a cosmic villain.',
        'A teenage Spider-Man tries to balance heroism and school.',
        'An older Wolverine protects a young mutant.',
        'An Amazon warrior discovers her powers during a world war.',
        'Two magicians compete to create the ultimate illusion.',
        'A freed slave sets out to rescue his wife.',
        'A drummer faces abuse in pursuit of greatness.',
        'An actress and pianist fall in love in L.A.',
        'A poor family infiltrates a rich household.',
        'Two soldiers race to stop a deadly attack during WWI.',
        'A frontiersman fights for survival after a bear attack.',
        'A secret agent manipulates time to stop a global threat.'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head(15))

# Optional: Save to CSV
df.to_csv("movies_50_dataset.csv", index=False)


# In[6]:


# Combine genres and description into one content field
df["content"] = df["genres"].fillna('') + " " + df["description"].fillna('')
df.head()


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Convert content into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(df["content"])


# In[9]:


from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)


# In[10]:


def recommend_movie(title):
    if title not in df['title'].values:
        print(f"Movie '{title}' not found.")
        return
    
    index = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first one (it’s the movie itself)
    top_5 = similarity_scores[1:6]
    
    print(f"\n🎬 Because you liked '{title}', you may also like:")
    for i in top_5:
        print(" -", df.iloc[i[0]]["title"])


# In[11]:


recommend_movie("Fight Club")


# In[12]:


recommend_movie("Tenet")


# In[13]:


import matplotlib.pyplot as plt

# Count all genres
genre_counts = df["genres"].str.split('|').explode().value_counts()

# Plot
genre_counts.plot(kind="bar", color="skyblue", title="Movie Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# In[ ]:




