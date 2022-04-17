def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(x):
    # find most likely label index for each element in the batch
    return x.argmax(dim=-1)

def confidence_interval(x):
    x = np.array(x)
    left, right = st.t.interval(alpha=0.95, df=len(x)-1, loc=np.mean(x), scale=st.sem(x)) 
    return np.mean(x), np.mean(x) -left