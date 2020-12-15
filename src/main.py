import os
from decouple import config
from src.utils.get_tweets_by_id import get_tweets_by_id


if __name__ == "__main__":

    df = get_tweets_by_id(config,
                          os.path.join('data', 'NAACL_SRW_2016.csv'))