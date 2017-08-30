# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:22:27 2017


"""

import kivy
import os
import pandas as pd
import numpy as np
import threading
from threading import Thread
from time import sleep

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.network.urlrequest import UrlRequest
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from algorithm import MedicalPredictor

import matplotlib.pyplot as plt
plt.style.use('dark_background')


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,target):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()




class MedicalWidget(GridLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def __init__(self):
        super(MedicalWidget, self).__init__(cols=2)
        self.predictor = None
        self.analyzedRecord = None
        self.fromFile = False
        self.method = ''

        self.secondThread = None
        self.stopThread = False

        self.dropdown = CustomDropDown()
        self.loadDialog = LoadDialog()
        self.data = []
        self.fig, self.ax = plt.subplots()
        canvas = self.fig.canvas
        plot1 = self.ax.plot()
        self.ids.plots.add_widget(canvas)
        self.ids.plots.add_widget(Label(text='Accuracy',  size_hint_y=None, height=20))
        self.ids.plots.add_widget(Label(text='', size_hint_y=None, height=20))
        self.ids.plots.add_widget(Label(text='', size_hint_y=None, height=20))
        self.ids.plots.add_widget(Label(text='', size_hint_y=None, height=50))

    def callback(self, id):
        print(str(id)+' method is being evaluated')

    def call_dropdown(self, caller):
        self.dropdown.open(caller)
        self.dropdown.bind(on_select=lambda instance, x: setattr(caller, 'text', x))

    def load(self, path, filename):
        self.fromFile = True
        try:
            with open(os.path.join(path, filename[0])) as stream:
                self.data = pd.read_csv(os.path.join(path, filename[0]), header=None)
                self.ids.clump.text = str(self.data[0][0])
                self.ids.cell_size.text = str(self.data[1][0])
                self.ids.cell_shape.text = str(self.data[2][0])
                self.ids.adhesion.text = str(self.data[3][0])
                self.ids.epithelial.text = str(self.data[4][0])
                self.ids.bare.text = str(self.data[5][0])
                self.ids.bland.text = str(self.data[6][0])
                self.ids.nucleoli.text = str(self.data[7][0])
                self.ids.mitose.text = str(self.data[8][0])
                self.ids.ID.text = str(self.data[9][0])
            self.analyzedRecord = self.data
        except:
            print('Choose file')

    def cancel(self):
        self.data = []
        self.ids.clump.text = ''
        self.ids.cell_size.text = ''
        self.ids.cell_shape.text = ''
        self.ids.adhesion.text = ''
        self.ids.epithelial.text = ''
        self.ids.bare.text = ''
        self.ids.bland.text = ''
        self.ids.nucleoli.text = ''
        self.ids.mitose.text = ''
        self.ids.ID.text = ''

        self.analyzedRecord = self.data

    def plot(self):
        self.predictor = MedicalPredictor(r'path_to_data')
        self.secondThread = Thread(target=self.worker)
        self.secondThread.start()

    def worker(self):
        if not self.fromFile:
            self.analyzedRecord = {0: self.ids.clump.text, 1: self.ids.cell_size.text, 2: self.ids.cell_shape.text,
                                   3: self.ids.adhesion.text, 4: self.ids.epithelial.text, 5: self.ids.bare.text,
                                   6: self.ids.bland.text, 7: self.ids.nucleoli.text, 8: self.ids.mitose.text,
                                   9: self.ids.ID.text}
            self.analyzedRecord = pd.DataFrame(self.analyzedRecord, index=[0])
            print(self.analyzedRecord)
        try:
            self.fig = self.predictor.PCA(self.analyzedRecord)
        except:
            print('Fill in all fields')
            return
        try:
            model = self.dropdown.show_method()
            print(model)
        except:
            print('Choose classifier!')
            return
        diagnosis, accuracy, report = self.predictor.classify(model)
        print(report)
        plt.draw()
        plt.show()
        if diagnosis[0] == 1:
            text = 'Benign\nID: ' + str(self.analyzedRecord[9][0])
        else:
            text = 'Malignant\nID: ' + str(self.analyzedRecord[9][0])
        self.ids.plots.children[0].text = text
        self.ids.plots.children[1].text = str(accuracy)
        #self.ids.plots.children[2].text = str(report)
        return

    def stop(self):
        self.secondThread.terminate()
        self.secondThread.join()


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class CustomDropDown(DropDown):
    def __init__(self):
        super(CustomDropDown, self).__init__()
        self.data = 'test'

    def on_select(self, data):
        self.data = data

    def show_method(self):
        return self.data


class MyApp(App):
    def build(self):
        return MedicalWidget()

if __name__ == '__main__':
    MyApp().run()
