# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:22:27 2017


"""

import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.widget import Widget

class MedicalWidget(Widget):
    pass


class MyApp(App):
    def build(self):
        return MedicalWidget()
if __name__=='__main__':
    MyApp().run()