from kivy.app import App
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import ObjectProperty
from kivy.properties import NumericProperty
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter

from oct2py import octave
import os

Builder.load_file('Screen1.kv')


class LoadImage(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class LoadObj(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class HomeScreen(Screen):
    styleimg = ObjectProperty(None)
    objimg = ObjectProperty(None)

    def load(self, path, filename):
        app=App.get_running_app()
        self.styleimg.source=filename[0]
        app.styleimage=filename[0]
        self.styleimg.reload()
        self.dismiss_popup()
    
    def show_load(self):
        content = LoadImage(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Image", content=content,size_hint=(0.9, 0.9))
        self._popup.open()

    def load_obj(self, path, filename):
        app=App.get_running_app()
        self.objimg.source=filename[0]
        app.objectimg=filename[0]
        self.objimg.reload()
        self.dismiss_popup()
    
    def show_loadobj(self):
        content = LoadObj(load=self.load_obj, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Object", content=content,size_hint=(0.9, 0.9))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

class Screen2(Screen):
    def generate_work_image(self,x):
        app=App.get_running_app()
        octave.f1(app.styleimage,app.objectimg,x[3],x[1],x[2],list(x[0]))
        print(x)
        app.newimg="newimg.png"
        app.sm.add_widget(OutputScreen(name="output"))
        self.manager.current="output"


        

class OutputScreen(Screen):
    pass


class MainApp(App):
    styleimage=StringProperty('images\download.jpg')
    objectimg=StringProperty('images\peacock.png')
    newimg=StringProperty("newimg.png")
    sm=ScreenManager()
    def build(self):
        self.sm.add_widget(HomeScreen(name="home"))
        self.sm.add_widget(Screen2(name="placingscreen"))
        return self.sm

if __name__ == "__main__":
    MainApp().run()