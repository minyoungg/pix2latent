

class TransformTemplate():

    def __init__(self):
        return


    def __call__(self):
        """ applies transformation to the image """
        raise NotImplementedError


    def get_default_param(self):
        """ applies transformation to the image """
        raise NotImplementedError


    def get_identity_param(self):
        """ applies identity transformation to the image """
        raise NotImplementedError


    def transform(self):
        """ applies transformation to the image """
        raise NotImplementedError


    def invert_transform(self):
        """ applies inverse transformation to the image """
        raise NotImplementedError
