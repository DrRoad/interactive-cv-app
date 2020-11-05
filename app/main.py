from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.utils import platform
from kivy.logger import Logger
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, DictProperty, BooleanProperty, ListProperty
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.metrics import dp
from kivy.core.text import LabelBase

from .widgets import *
from .screens import *

import cv2 as cv
import numpy as np
from copy import deepcopy


DEFAULT_SETTINGS = {
    "Canny": {'th1': 100, 'th2': 150},
    "Threshold": {'threshold': 125, "otsu": False},
    "AdapThreshold": {'block_size': 3, "c_size": 0, 'filter_type': 'box'},
    "Blur": {'kernel_size': 3},
    "OpticalFlow": {'window_size': 25, 'min_speed_threshold': 2},
    "Convolution": {'kernel': 'Sobel (left)', 'scale_factor': 1}
    }

CONVOLUTION_KERNELS = {'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                       'Sobel (left)': np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
                       'Sobel (right)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                       'Sobel (up)': np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
                       'Sobel (down)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                       'Blur': np.array([[.0625, .125, .0625], [.125, .25, .125], [.0625, .125, .0625]]),
                       "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
                       "Outline": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])}

LabelBase.register(name="BebasNeue",
                   fn_regular="fonts/BebasNeue-Regular.ttf")

Builder.load_string("""

<RootLayout>:
    kivycam: kivy_cam
    manager: manager
    KivyCam:
        id: kivy_cam
        size_hint: (1, 1)
    FilterManager:
        id: manager
        size_hint: (1, .25)
        pos_hint: {'x': 0}

<FilterManager>:
    id: manager
    MainScreen:
        id: main_screen
        name: 'main_screen'
        manager: manager
    CannyScreen:
        id: canny_screen
        name: 'canny_screen'
        manager: manager
    ThresholdScreen:
        id: binary_thresh_screen
        name: 'binary_thresh_screen'
        manager: manager
    AdapThreshScreen:
        id: adap_thresh_screen
        name: 'adap_thresh_screen'
        manager: manager
    BlurScreen:
        id: blur_screen
        name: 'blur_screen'
        manager: manager
    OpticalFlowScreen:
        id: optical_flow_screen
        name: 'optical_flow_screen'
        manager: manager
    ConvolutionScreen:
        id: convolution_screen
        name: 'convolution_screen'
        manager: manager

<Header>:
    size_hint: (1, .2)
    pos_hint: {'center_x': .5, 'top': 1}
    Button:
        text: "<"
        bold: True
        size_hint: (None, 1)
        width: self.height
        text_size: self.size
        font_size: .75*self.height
        halign: 'center'
        valign: 'top'
        pos_hint: {'x': 0, 'top': 1}
        background_color: .4, .4, .4, 0
        on_release: app.root.manager.change_screen(name="main_screen", transition="right")
    Label:
        text: root.title
        pos_hint: {'center_x': .5, 'top': 1}
        size_hint: (1, 1)
        text_size: self.size
        font_size: .7*self.height
        halign: 'center'
        valign: 'top'
""")
Builder.load_file('app/kv/screens.kv')
Builder.load_file('app/kv/widgets.kv')
                    
class Header(RelativeLayout):
    title = StringProperty()
    

class RootLayout(FloatLayout):
    kivycam = ObjectProperty()
    manager = ObjectProperty()


class FilterManager(ScreenManager):

    def change_screen(self, name, transition):
        self.transition.direction = transition
        self.current = name


class KivyCam(Image):

    active_filter = StringProperty(None, allownone=True)
    settings = DictProperty()
    prevImg = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv.VideoCapture(0)
        self.default_settings()
        Clock.schedule_interval(self.update, 1.0/10)
        
    def crop_frame(self, frame):
        window_width, window_height = Window.size
        frame_height, frame_width, _ = frame.shape
        scale_factor = window_height/frame_height
        new_frame_shape = (int(scale_factor*frame_width), window_height)
        new_frame = cv.resize(frame, dsize=new_frame_shape)
        x0 = (new_frame_shape[0] - window_width)//2
        return new_frame[:, x0:x0+window_width]

    def update(self, dt):

        ret, raw_frame = self.capture.read()

        if ret:

            raw_frame = cv.flip(raw_frame, 0)
            raw_frame = cv.flip(raw_frame, 1)
            if platform == 'macosx':
                raw_frame = self.crop_frame(raw_frame)
            frame = raw_frame.copy()

            if self.prevImg is None:
                self.prevImg = frame

            if self.active_filter == "Canny":
                th1 = self.settings["Canny"]["th1"]
                th2 = self.settings["Canny"]["th2"]
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.Canny(frame, th1, th2)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            elif self.active_filter == "Threshold":
                th = self.settings["Threshold"]["threshold"]
                otsu_enabled = self.settings["Threshold"]['otsu']
                type_ = cv.THRESH_OTSU if otsu_enabled else cv.THRESH_BINARY
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                retval, frame = cv.threshold(frame, th, 255, type_)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            elif self.active_filter == "AdapThreshold":
                block_size = self.settings["AdapThreshold"]["block_size"]
                c = self.settings["AdapThreshold"]["c_size"]
                filter_type = cv.ADAPTIVE_THRESH_MEAN_C if self.settings["AdapThreshold"]["filter_type"] == "box" else cv.ADAPTIVE_THRESH_GAUSSIAN_C
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.adaptiveThreshold(src=frame,
                                             maxValue=255,
                                             adaptiveMethod=filter_type,
                                             thresholdType=cv.THRESH_BINARY,
                                             blockSize=block_size,
                                             C=c)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            elif self.active_filter == "Blur":
                kernel_size = self.settings["Blur"]["kernel_size"]
                frame = cv.blur(frame, ksize=(kernel_size, kernel_size))
            elif self.active_filter == "OpticalFlow":
                prevImg = cv.cvtColor(self.prevImg, cv.COLOR_BGR2GRAY)
                nextImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                wsize = self.settings["OpticalFlow"]["window_size"]
                min_speed = self.settings["OpticalFlow"]["min_speed_threshold"]
                flow = cv.calcOpticalFlowFarneback(prev=prevImg,
                                                   next=nextImg,
                                                   flow=None,
                                                   pyr_scale=0.5,
                                                   levels=3,
                                                   winsize=wsize,
                                                   iterations=3,
                                                   poly_n=5,
                                                   poly_sigma=1.1,
                                                   flags=0)

                flow_smoothed = np.zeros_like(flow)
                flow_smoothed[:,:,0] = cv.GaussianBlur(flow[:,:,0], ksize=(wsize,wsize), sigmaX=wsize, sigmaY=wsize)
                flow_smoothed[:,:,1] = cv.GaussianBlur(flow[:,:,1], ksize=(wsize,wsize), sigmaX=wsize, sigmaY=wsize)
                flow = flow_smoothed

                mag, angle = cv.cartToPolar(flow[:, :, 0],
                                            flow[:, :, 1],
                                            angleInDegrees=True)
                mag[mag < min_speed] = 0

                hsv_mask = np.zeros_like(frame)
                hsv_mask[:, :, 0] = angle//2
                hsv_mask[:, :, 1] = 255
                hsv_mask[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

                frame = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)

            elif self.active_filter == "Convolution":
                kernel = CONVOLUTION_KERNELS[self.settings["Convolution"]["kernel"]]
                scale = self.settings["Convolution"]["scale_factor"]
                
                old_size = frame.shape[1], frame.shape[0]
                new_size = frame.shape[1]//scale, frame.shape[0]//scale
                frame = cv.resize(src=frame, dsize=new_size)
                
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.filter2D(src=frame, ddepth=-1, kernel=kernel)
                frame = cv.resize(src=frame, dsize=old_size)
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

            self.prevImg = raw_frame

            # Convert array to image texture
            img = frame.tobytes()
            tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

            # This line fails
            tex.blit_buffer(img, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = tex
        else:
            Logger.error("Could not access device's camera.")

    def default_settings(self):
        self.settings = deepcopy(DEFAULT_SETTINGS)
        self.active_filter = None


class CameraApp(App):

    FONT = StringProperty()
    FONT_SIZE = NumericProperty()

    def detect_platform(self):
        """Detects the operating system the app is being run on. If on a
            desktop, adjusts the screen size to simulate a
            phone screen (for debugging purposes)"""

        platforms = {'macosx': 'Mac OSX', 'win': 'Windows', 'linux': 'Linux',
                     'ios': 'iOS', 'android': 'Android'}

        if platform in ('macosx', 'linux', 'win'):
            Window.size = (240, 490)
        elif platform == 'ios':
            Logger.warning('This app is untested on iOS and may not work.')
        else:
            Logger.warning('Did not recognize OS. Attempting to run app anyway.')
        Logger.info('System detected: {}'.format(platforms[platform]))

    def build(self):

        self.FONT = 'BebasNeue'
        self.FONT_SIZE = int(.025*Window.size[1])

        return RootLayout()
