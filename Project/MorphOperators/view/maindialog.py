__author__ = "Haim Adrian"

import os
import tkinter as tk
import tkinter.ttk as ttk
from threading import Thread
from tkinter import messagebox
from tkinter.ttk import Style

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import view.controls as ctl
from logic.objectdetectionlogic import runObjectDetection
from util.settings import Settings
from view.fileinput import FileInput
from view.settingsdialog import SettingsDialog
from view.tooltip import Tooltip

TITLE = 'Morphological Object Detector'


class MainDialog(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        master.config(background=ctl.BACKGROUND_COLOR)
        toplevel = self.winfo_toplevel()
        toplevel.title(TITLE)
        toplevel.iconbitmap(os.path.abspath(os.path.join('resource', 'corners-icon.ico')))

        # Declare all of the instance attributes here, and initialize them in separate methods for
        # code separation.
        # Keep them as attributes so we will not have issues with garbage collection.
        # Especially with the PhotoImage
        self.settings = Settings().load()
        self.titleFrame = None  # tk.Frame
        self.actionFrame = None  # tk.Frame
        self.statusFrame = None  # tk.Frame
        self.figureFrame = None  # tk.Frame
        self.title = None  # tk.Label
        self.statusBar = None  # tk.Label
        self.progressBar = None  # tk.Progressbar
        self.settingsButton = None  # tk.Button
        self.goButton = None  # tk.Button
        self.openObj1FileButton = None  # tk.Button
        self.popupImageButton = None  # tk.Button
        self.settingsButtonTooltip = None  # view.tooltip.Tooltip
        self.goButtonTooltip = None  # view.tooltip.Tooltip
        self.popupImageButtonTooltip = None  # view.tooltip.Tooltip
        self.filePathObj1Entry = None  # tk.Entry
        self.filePathObj2Entry = None  # tk.Entry
        self.filePathImageEntry = None  # tk.Entry
        self.magnifyingIcon = None  # tk.PhotoImage
        self.playIcon = None  # tk.PhotoImage
        self.popoutIcon = None  # tk.PhotoImage
        self.settingsIcon = None  # tk.PhotoImage
        self.figure = None  # A reference to pyplot figure, so we can destroy it
        self.style = None  # tk.ttk.Style
        self.isRunning = False  # Indication of when we wait for the worker to finish
        self.obj1 = None  # A reference to the input image. It is being set by the action
        self.obj2 = None  # A reference to the input image. It is being set by the action
        self.image = None  # A reference to the input image. It is being set by the action
        self.objectsImage = None  # A reference to the processed image (outcome).
        self.objectsBinaryImage = None  # A reference to the processed image (outcome).
        self.objectsClosingImage = None  # A reference to the processed image (outcome).
        self.imageBinary = None  # A reference to the processed image (outcome).
        self.imageClosing = None  # A reference to the processed image (outcome).
        self.hitMissObj1 = None  # A reference to the processed image (outcome).
        self.hitMissObj2 = None  # A reference to the processed image (outcome).
        self.imageMarks = None  # A reference to the processed image (outcome).
        self.error = False  # Indication for a failure during algorithm
        self.progressTextFormat = '{0}%'

        self.style = Style(master)
        self.style.theme_use('clam')
        self.style.configure("Horizontal.TProgressbar",
                             foreground='#E0E0E0',
                             background=ctl.ACCEPT_COLOR,
                             troughcolor=ctl.BACKGROUND_TOOLTIP_COLOR,
                             bordercolor=ctl.ACCEPT_COLOR,
                             lightcolor=ctl.ACCEPT_COLOR,
                             darkcolor=ctl.ACCEPT_COLOR)
        self.style.configure('TButton',
                             font=ctl.FONT_BUTTON,
                             borderwidth='1',
                             background=ctl.BACKGROUND_COLOR,
                             relief='FLAT')
        self.style.configure('TLabel', font=ctl.FONT_REGULAR)
        self.style.map('TButton', background=[('active', '!disabled', '#4E6067')])
        self.style.map('TCombobox', foreground=[('readonly', '!disabled', ctl.FOREGROUND_EDITOR_COLOR)])
        self.style.map('TCombobox', background=[('readonly', '!disabled', ctl.BACKGROUND_EDITOR_COLOR)])
        self.style.map('TCombobox', fieldbackground=[('readonly', '!disabled', ctl.BACKGROUND_EDITOR_COLOR)])
        self.style.map('TCombobox', lightcolor=[('readonly', '!disabled', 'black')])
        self.style.map('TCombobox', darkcolor=[('readonly', '!disabled', 'black')])
        self.style.map('TCombobox', bordercolor=[('readonly', '!disabled', '#E0E0E0')])
        master.option_add("*TCombobox*Listbox*Background", ctl.BACKGROUND_EDITOR_COLOR)
        master.option_add("*TCombobox*Listbox*Foreground", ctl.FOREGROUND_EDITOR_COLOR)

        self.createTitleSection(master)
        self.createActionSection(master)
        self.createStatusSection(master)
        self.createWorkAreaSection(master)

        master.bind('<Return>', self.onEnterPressed)
        master.geometry("1000x800")

    def createTitleSection(self, master):
        """
        Title area contains a frame with a label in it.
        :param master: Master dialog to add the frame to
        :return: None
        """
        self.titleFrame = ctl.frame(master, tk.X)
        self.title = ctl.title(self.titleFrame, TITLE)
        self.title.pack(fill=tk.X)

    def createActionSection(self, master):
        """
        Action area contains a frame with an entry (text edit for image path), open file dialog
        button and Go button.
        Open file dialog button will display an open file dialog to select image
        Go button will execute Harris Detector action on the selected image
        :param master: Master dialog to add the frame to
        :return: None
        """
        self.actionFrame = ctl.frame(master, tk.X)
        self.magnifyingIcon = tk.PhotoImage(file=os.path.abspath(
            os.path.join('resource', 'magnifying-icon.png')))
        self.filePathObj1Entry = FileInput(self.actionFrame, "Select First Object")
        self.filePathObj1Entry.pack(fill=tk.X, side=tk.TOP, expand=False)
        self.filePathObj2Entry = FileInput(self.actionFrame, "Select Second Object")
        self.filePathObj2Entry.pack(fill=tk.X, side=tk.TOP, expand=False)
        self.filePathImageEntry = FileInput(self.actionFrame, "Select Image or Video")
        self.filePathImageEntry.pack(fill=tk.X, side=tk.TOP, expand=False)

        # Get a second line in the actions area, so the checkbox will be under the entry
        helper_frame = ctl.frame(master, tk.X)

        self.playIcon = tk.PhotoImage(file=os.path.abspath(
            os.path.join('resource', 'play-icon.png')))
        self.goButton = ttk.Button(master=helper_frame,
                                   image=self.playIcon,
                                   command=self.onEnterPressed,
                                   style='TButton',
                                   width=4)
        self.goButtonTooltip = Tooltip(self.goButton, 'Run')
        self.goButton.pack(side=tk.RIGHT)

        self.settingsIcon = tk.PhotoImage(file=os.path.abspath(
            os.path.join('resource', 'settings-icon.png')))
        self.settingsButton = ttk.Button(master=helper_frame,
                                         image=self.settingsIcon,
                                         command=self.showSettingsDialog,
                                         style='TButton')
        self.settingsButtonTooltip = Tooltip(self.settingsButton, 'Settings')
        self.settingsButton.pack(side=tk.RIGHT)

        self.popoutIcon = tk.PhotoImage(file=os.path.abspath(
            os.path.join('resource', 'pop-out-icon.png')))
        self.popupImageButton = ttk.Button(master=helper_frame,
                                           image=self.popoutIcon,
                                           command=self.popupImage,
                                           style='TButton')
        self.popupImageButtonTooltip = Tooltip(self.popupImageButton,
                                               'Pops out the image with as a pyplot dialog')
        self.popupImageButton.pack(side=tk.LEFT)
        self.popupImageButton['state'] = 'disabled'
        helper_frame.pack(fill=tk.X, side=tk.TOP, expand=False)

    def createStatusSection(self, master):
        """
        Status area contains a progress bar which we put at the bottom of the frame, to show a
        progress indication while the algorithm is executing. (Can take some seconds)
        We also add a label into the progress bar to use it as a status bar as well
        :param master: Master dialog to add the frame to
        :return: None
        """
        self.statusFrame = ctl.frame(master, tk.X, 0, 0)
        self.statusFrame.pack(fill=tk.X, side=tk.BOTTOM, anchor=tk.S)
        # Add label into the layout
        self.style.layout('text.Horizontal.TProgressbar',
                          [('Horizontal.Progressbar.trough',
                            {
                                'children': [('Horizontal.Progressbar.pbar',
                                              {'side': 'left', 'sticky': 's'})],
                                'sticky': 'swe'
                            }),
                           ('Horizontal.Progressbar.label', {'sticky': 'we'})])
        self.style.configure('text.Horizontal.TProgressbar', font=ctl.FONT_REGULAR)
        self.progressBar = tk.ttk.Progressbar(master=self.statusFrame,
                                              orient=tk.HORIZONTAL,
                                              style='text.Horizontal.TProgressbar',
                                              length=101,
                                              mode='determinate',
                                              value=0,
                                              maximum=101)
        self.progressBar.pack(fill=tk.X)
        self.updateStatus('Open an image using the magnifying button and click Play')

    def createWorkAreaSection(self, master):
        """
        WorkArea contains a frame onto which we are displaying the plots when we finish executing
        :param master: Master dialog to add the frame to
        :return: None
        """
        self.figureFrame = ctl.frame(master, tk.BOTH)

        # Make it white because I could not modify the PyPlot background
        self.figureFrame.configure(background='white')

    def updateStatus(self, text):
        """
        Updates the text of the progress bar (status bar) with the specified text
        :param text: The text to set into the status bar
        :return: None
        """
        self.style.configure('text.Horizontal.TProgressbar', text=' ' + text)
        self.progressTextFormat = ' ' + text + ' {0}%'
        print(text)

    def updateProgress(self, progress):
        """
        Updates the progress bar with a progress, so user can see how much time remains
        :param progress: The progress to set to the progressbar
        :return: None
        """
        progress = np.min([100, int(progress)])
        self.progressBar['value'] = progress
        self.style.configure('text.Horizontal.TProgressbar',
                             text=self.progressTextFormat.format(progress))

    def startProgress(self):
        """
        Prepare for starting a progress in the progress bar, for visualizing process.
        We also start checking the job queue to detect if the algorithm has finished its execution
        so we can display the outcome
        :return: None
        """
        self.setUserComponentsState('disabled')
        self.image = self.obj1 = self.obj2 = self.objectsImage = self.objectsBinaryImage = \
            self.objectsClosingImage = self.imageBinary = self.imageClosing = self.imageMarks = None
        self.isRunning = True
        self.periodicallyCheckOutcome()

        if self.figure is not None:
            self.figure.set_visible(False)
            self.figure.clear()
            self.figure = None

            # Remove all children so we will not have issues with pyplot's painting
            for child in self.figureFrame.winfo_children():
                child.destroy()

    def stopProgress(self):
        """
        Stops the progress of the progress bar, and the periodic check of the job queue
        :return: None
        """
        self.progressBar['value'] = 0
        self.isRunning = False
        self.setUserComponentsState('normal')

    def setUserComponentsState(self, new_state):
        """
        Sets the state of user components to 'normal' or 'disabled'.
        When the algorithm is running in background, we disable the user components so there
        won't be a mess
        :param new_state: The new state to set. Can be one of 'normal' or 'disabled'
        :return: None
        """
        self.goButton['state'] = new_state
        self.popupImageButton['state'] = new_state
        self.filePathObj1Entry.setState(new_state)
        self.filePathObj2Entry.setState(new_state)
        self.filePathImageEntry.setState(new_state)

    def showSettingsDialog(self):
        """
        Displaying Settings dialog so user can customize the algorithm settings
        :return: None
        """
        settings = SettingsDialog(self.master).result
        if settings:
            self.settings = settings
            self.settings.save()

    def onEnterPressed(self, event=None):
        """
        This event is raised whenever user presses the Go button or Enter
        In this case we are going to execute Harris Detector algorithm using the selected image, in
        case there is a valid selection.
        :param event: The keypress event (We register on the master frame)
        :return: None
        """

        def isValidFile(filePath):
            if filePath == '':
                messagebox.showerror('Illegal Input', 'Please select a file first')
                return False
            if not os.path.exists(filePath) or not os.path.isfile(filePath):
                messagebox.showerror('Illegal Input',
                                     'Selected file is not a file or it does not exist:\n{}'
                                     .format(filePath))
                return False
            return True

        obj1FilePath = self.filePathObj1Entry.getSelectedFilePath()
        obj2FilePath = self.filePathObj2Entry.getSelectedFilePath()
        imageFilePath = self.filePathImageEntry.getSelectedFilePath()
        if isValidFile(obj1FilePath) and isValidFile(obj2FilePath) and isValidFile(imageFilePath):
            if not self.isRunning:
                self.startProgress()

                # Run it in background so the progress bar will not get blocked.
                # (We cannot block te gui thread)
                t = Thread(target=self.executeObjectCounterUsingMorphOperator, args=(obj1FilePath,
                                                                                     obj2FilePath,
                                                                                     imageFilePath))
                t.start()
            else:
                print('WARN - Already running. Cannot run multiple detections in parallel.')

    def executeObjectCounterUsingMorphOperator(self, obj1FilePath, obj2FilePath, imageFilePath):
        """
        The job of executing Harris Detector algorithm.
        It takes time so we have a specific action for that, such that we can run it using a
        background thread
        :param obj1FilePath: Path to the selected first object
        :param obj2FilePath: Path to the selected second object
        :param imageFilePath: Path to the selected image
        :return: None
        """
        self.obj1 = cv2.imread(obj1FilePath)
        self.obj2 = cv2.imread(obj2FilePath)
        self.image = cv2.resize(cv2.imread(imageFilePath), self.settings.imageShape)

        # To make the line shorter.........
        a, b, c, d, e, f, g, h = runObjectDetection(self.obj1,
                                                    self.obj2,
                                                    self.image,
                                                    self.settings,
                                                    lambda text: self.updateStatus(text),
                                                    lambda progress: self.updateProgress(progress))
        self.objectsImage = a
        self.objectsBinaryImage = b
        self.objectsClosingImage = c
        self.imageBinary = d
        self.imageClosing = e
        self.hitMissObj1 = f
        self.hitMissObj2 = g
        self.imageMarks = h

        if self.image is None or self.objectsImage is None:
            self.error = True

        self.stopProgress()

    def showImages(self):
        """
        When Harris Detector job has finished we display the results as embedded figure
        :return: None
        """
        self.figure = Figure(figsize=(5, 5), dpi=100)
        axes = self.figure.add_subplot(241, title='Objects')
        axes.axis("off")
        axes.imshow(np.uint8(self.objectsImage[:, :, ::-1]))  # -1 means BGRtoRGB
        axes = self.figure.add_subplot(242, title='Objects Binary')
        axes.axis("off")
        axes.imshow(np.uint8(self.objectsBinaryImage), cmap="gray")
        axes = self.figure.add_subplot(243, title='Objects Closing (Morph)')
        axes.axis("off")
        axes.imshow(np.uint8(self.objectsClosingImage), cmap="gray")
        axes = self.figure.add_subplot(244, title='Obj1 Hit&Miss in Image')
        axes.axis("off")
        axes.imshow(np.uint8(self.hitMissObj1), cmap="gray")
        axes = self.figure.add_subplot(245, title='Image Marks')
        axes.axis("off")
        axes.imshow(np.uint8(self.imageMarks[:, :, ::-1]))  # -1 means BGRtoRGB
        axes = self.figure.add_subplot(246, title='Image Binary')
        axes.axis("off")
        axes.imshow(np.uint8(self.imageBinary), cmap="gray")
        axes = self.figure.add_subplot(247, title='Image Closing (Morph)')
        axes.axis("off")
        axes.imshow(np.uint8(self.imageClosing), cmap="gray")
        axes = self.figure.add_subplot(248, title='Obj2 Hit&Miss in Image')
        axes.axis("off")
        axes.imshow(np.uint8(self.hitMissObj2), cmap="gray")

        self.figure.subplots_adjust(0.05, 0.1, 0.95, 0.9, 0.2, 0.25)
        canvas = FigureCanvasTkAgg(self.figure, self.figureFrame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.figureFrame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def periodicallyCheckOutcome(self):
        """
        Check every 200 ms if there is something new to show, which means a worker
        has finished and we can pick the output and show it in the GUI
        """
        if self.error:
            self.error = False
            self.stopProgress()
            messagebox.showerror('Error', 'Error has occurred while trying to detect corners')
            self.image = self.objectsImage = None
            return None

        if self.image is not None and self.objectsImage is not None:
            self.stopProgress()
            # Plot the images, embedded within our dialog rather than popping up another dialog.
            self.showImages()

        if self.isRunning:
            self.master.after(50, self.periodicallyCheckOutcome)

    def popupImage(self):
        """
        Action corresponding to when user presses the popup button, to plot the outcome outside the
        main window
        :return: None
        """
        if self.imageMarks is not None:
            plt.figure(TITLE)
            plt.tight_layout(pad=0.5)
            plt.imshow(np.uint8(self.imageMarks[:, :, ::-1]))
            plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.1, 0.1)
            figure_manager = plt.get_current_fig_manager()
            figure_manager.window.iconbitmap(os.path.abspath(os.path.join('resource', 'corners-icon.ico')))
            try:
                # In case there would be an issue on Ubuntu or something like that,
                # just catch the error and use resize
                figure_manager.window.state('zoomed')
            except Exception as e:
                print('ERROR - Error has occurred while trying to show window as maximized. ' +
                      'Fallback to resize strategy:', str(e))
                figure_manager.resize(*figure_manager.window.maxsize())
                figure_manager.window.wm_geometry("+0+0")

            plt.show()
