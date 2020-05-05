import pandas as pd
from tqdm import tqdm

outfile = 'data/yelp_restaurant_reviews.csv'

# read in business data
business_json_path = './data/yelp_academic_dataset_business.json'
df_b = pd.read_json(business_json_path, lines=True)
df_b = df_b[df_b['is_open']==1] # open businesses only

# we only care about the business ID
drop_columns = ['name','address','city','state','hours',
                'latitude','longitude', 'stars', 'review_count',
                'is_open', 'attributes', 'hours', 'postal_code']
df_b = df_b.drop(drop_columns, axis=1)

# only keep the restaurants
df_restaurants = df_b[df_b['categories'].str.contains(
              'Restaurant|Restaurants|Food|Pub|Bar',
              case=False, na=False)]

df_restaurants = df_restaurants.drop(['categories'], axis=1)

review_json_path = './data/yelp_academic_dataset_review.json'

size = 100000
review = pd.read_json(review_json_path, lines=True,
                      dtype={'review_id':str,'user_id':str,
                             'business_id':str,'stars':int,
                             'date':str,'text':str,'useful':int,
                             'funny':int,'cool':int},
                     chunksize=size)

# There are multiple chunks to be read
chunk_list = []
for chunk_review in tqdm(review):
    # Drop columns that aren't needed
    chunk_review = chunk_review.drop(['useful','funny','cool'], axis=1)
    # Inner merge with edited business file so only reviews related to the business remain
    chunk_merged = pd.merge(df_restaurants, chunk_review, on='business_id', how='inner')
    # Show feedback on progress
    print(f"{chunk_merged.shape[0]} out of {size:,} related reviews")
    chunk_list.append(chunk_merged)

# After trimming down the review file, concatenate all relevant data back to one dataframe
df = pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)
df.to_csv(outfile, index=False)
