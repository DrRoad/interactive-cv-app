from kivy.uix.screenmanager import Screen
from kivy.properties import NumericProperty, BooleanProperty, StringProperty
from kivy.app import App


class ScreenTemplate(Screen):
    pass


class MainScreen(ScreenTemplate):
    pass


class CannyScreen(ScreenTemplate):
    threshold1 = NumericProperty()
    threshold2 = NumericProperty()

    def on_pre_enter(self):
        app = App.get_running_app()
        self.threshold1 = app.root.kivycam.settings["Canny"]["th1"]
        self.threshold2 = app.root.kivycam.settings["Canny"]["th2"]

    def on_enter(self):
        app = App.get_running_app()
        app.root.kivycam.active_filter = "Canny"


class ThresholdScreen(ScreenTemplate):
    threshold = NumericProperty()
    otsu_enabled = BooleanProperty()

    def on_pre_enter(self):
        app = App.get_running_app()
        self.threshold = app.root.kivycam.settings["Threshold"]["threshold"]
        self.otsu_enabled = app.root.kivycam.settings["Threshold"]["otsu"]

    def on_enter(self):
        app = App.get_running_app()
        app.root.kivycam.active_filter = "Threshold"


class AdapThreshScreen(ScreenTemplate):
    block_size = NumericProperty()
    c_size = NumericProperty()

    def on_pre_enter(self):
        app = App.get_running_app()
        self.block_size = app.root.kivycam.settings["AdapThreshold"]["block_size"]
        self.c_size = app.root.kivycam.settings["AdapThreshold"]["c_size"]

    def on_enter(self):
        app = App.get_running_app()
        app.root.kivycam.active_filter = "AdapThreshold"


class BlurScreen(ScreenTemplate):
    kernel_size = NumericProperty()

    def on_pre_enter(self):
        app = App.get_running_app()
        self.kernel_size = app.root.kivycam.settings["Blur"]["kernel_size"]

    def on_enter(self):
        app = App.get_running_app()
        app.root.kivycam.active_filter = "Blur"


class OpticalFlowScreen(ScreenTemplate):
    window_size = NumericProperty()
    min_speed_threshold = NumericProperty()

    def on_pre_enter(self):
        app = App.get_running_app()
        self.window_size = app.root.kivycam.settings["OpticalFlow"]["window_size"]
        self.min_speed_threshold = app.root.kivycam.settings["OpticalFlow"]["min_speed_threshold"]

    def on_enter(self):
        app = App.get_running_app()
        app.root.kivycam.active_filter = "OpticalFlow"


class ConvolutionScreen(ScreenTemplate):
    kernel_size = NumericProperty()
    kernel = StringProperty()

    def on_pre_enter(self):
        app = App.get_running_app()
        self.kernel = app.root.kivycam.settings["Convolution"]["kernel"]
        self.scale_factor = app.root.kivycam.settings["Convolution"]["scale_factor"]
        #self.min_speed_threshold = app.root.kivycam.settings["OpticalFlow"]["min_speed_threshold"]

    def on_enter(self):
        app = App.get_running_app()
        app.root.kivycam.active_filter = "Convolution"
        
