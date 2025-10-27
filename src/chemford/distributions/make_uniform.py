from chemford.distributions.make_multinomial import make_multinomial

def make_uniform(random_state=None):
    return make_multinomial(list(range(1,10,1)), [1/9] * 9,random_state=random_state)