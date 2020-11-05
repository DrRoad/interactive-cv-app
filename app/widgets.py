from kivy.properties import ObjectProperty, ListProperty, NumericProperty, BooleanProperty, StringProperty, DictProperty
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.uix.behaviors import ToggleButtonBehavior, ButtonBehavior


class SliderBlock(RelativeLayout):
    value = NumericProperty()
    title = StringProperty()
    disabled = BooleanProperty(False)
    step = NumericProperty()


class SwitchBlock(RelativeLayout):
    active = BooleanProperty()
    title = StringProperty()


class RoundedToggleButton(ToggleButtonBehavior, Label):
    background_normal = ListProperty()
    background_down = ListProperty()
    radius = NumericProperty()
    

class RoundedButton(ButtonBehavior, Label):
    background_normal = ListProperty()
    background_down = ListProperty()
    radius = NumericProperty()

class KernelCreator:
    """Contains functions used to generate convolution kernels of arbitrary size"""
    
    def sobel_left(self, size):
        mid = size // 2