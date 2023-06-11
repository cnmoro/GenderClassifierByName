from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import pickle

dataset_path_female = 'BrazilFemaleUTF8.csv'
dataset_path_male = 'BrazilMaleUTF8.csv'

def clean_name(name):
    # Remove all characters that are not letters from A to Z
    name = ''.join([c for c in name if c.isalpha()])
    # Convert to lowercase
    return name.lower()

female_names = []
# Read the data, each line contains a name
with open('BrazilFemaleUTF8.csv', 'r', encoding='utf-8') as f:
    female_names = f.readlines()
female_names = [clean_name(fn.strip()) for fn in female_names]

male_names = []
# Read the data, each line contains a name
with open('BrazilMaleUTF8.csv', 'r', encoding='utf-8') as f:
    male_names = f.readlines()
male_names = [clean_name(fn.strip()) for fn in male_names]


# Function to extract new features
def extract_features(names):
    return [[name[-1], len(name)] for name in names]

# Custom transformer to apply the function
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return extract_features(data)
    
# Use ColumnTransformer to handle the different data types
preprocess = make_column_transformer(
    (OneHotEncoder(), [0]),  # apply OneHotEncoder to the first column (last letter)
    remainder='passthrough'  # leave the rest of the columns (length of the name)
)

# Build the feature union
features = FeatureUnion([
    ('ngram', CountVectorizer(analyzer='char', ngram_range=(2, 2))),
    ('last_letter_and_length', make_pipeline(FeatureExtractor(), preprocess))
])

# Make a pipeline: feature extraction -> classifier
model = make_pipeline(features, LogisticRegression())

# Prepare data for training
names = female_names + male_names
labels = ['female'] * len(female_names) + ['male'] * len(male_names)

# Train the model
model.fit(names, labels)

# Save model to file
pickle.dump(model, open('model.pkl', 'wb'))

# Make a prediction for a new name
print(model.predict(['Carlo'])[0]) # outputs 'male'

# Load model to make new predictions
model = pickle.load(open('model.pkl', 'rb'))
