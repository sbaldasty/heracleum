class Attack:

    def poison_data(some_params):
        pass

    def poison_gradients(some_params):
        pass


class SignFlipAttack(Attack):
    '''
    A malicious client flips the sign of their local update.
    '''
    pass


class ScalingAttack(Attack):
    '''
    A malicious client scales their local update by some value.
    '''
    pass