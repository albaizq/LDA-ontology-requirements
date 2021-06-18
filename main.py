import pandas as pd

import modules.dataDescription
import modules.topicExtraction


# The goal of these functions is to obtain the topics related to a set of ontology requirements.
if __name__ == '__main__':
    df = pd.read_csv(r"\dataset\requirements.csv",
                     sep=r'\s*;\s*', header=0, encoding='utf-8', engine='python')
    reqs = modules.dataDescription.analyse_data(df)
    modules.topicExtraction.get_topics(reqs)
