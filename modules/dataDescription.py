
# Data cleaning
def remove_duplicates(df):
    """This function will remove duplicated rows"""
    no_duplicates = df.drop_duplicates(subset=['Requirement'], keep='first')
    return no_duplicates


def count_duplicates(df, column):
    """This function will count duplicates"""
    return df.groupby([column]).size().reset_index(name='counts').sort_values('counts', ascending=False)


def analyse_data(df):
    print("1) Number of requirements")
    print(df.count())
    print("--------------------------------")

    print("2) Repeated requirements")
    reqs = count_duplicates(df, 'Requirement')
    print(reqs)
    print("--------------------------------")
    print("Action - Duplicates removed")
    print("--------------------------------")
    return remove_duplicates(df)
