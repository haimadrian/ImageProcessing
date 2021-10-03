__author__ = "Haim Adrian"

import os
import tkinter as tk
from ast import literal_eval
from tkinter import messagebox

import numpy as np

import view.controls as ctl
from util.settings import settingsInstance


def resetText(widget, value):
    widget.delete(0, tk.END)
    widget.insert(0, str(value))


def isNumeric(x):
    import re

    # Store a compiled regex onto the function so we will not have to recompile it over and over
    if not hasattr(isNumeric, 'numericRegex'):
        # Do not accept negative values (define + in the regex only, without -)
        isNumeric.numericRegex = re.compile(r"^([+-]?\d*)\.?\d*$")
    return len(str(x).strip()) > 0 and isNumeric.numericRegex.match(str(x).strip()) is not None


def isTuple(x):
    import re

    # Store a compiled regex onto the function so we will not have to recompile it over and over
    if not hasattr(isTuple, 'tupleRegex'):
        isTuple.tupleRegex = re.compile(r"^ *\( *\d+ *, *\d+ *, *\d+ *\) *$")
    return len(str(x).strip()) > 0 and isTuple.tupleRegex.match(str(x).strip()) is not None


def numericValidator(widget, oldText, newText):
    """
    A function used to validate the input of an entry, to make sure it is numeric
    :param widget: Sender widget
    :param oldText: Input text before the change
    :param newText: Input text
    :return: Whether text is a valid number or not
    """
    isValid = isNumeric(newText) or newText == ''

    if not isValid:
        messagebox.showerror('Illegal Input', 'Input must be numeric. Was: {}'.format(newText))
        widget.delete(0, tk.END)
        widget.insert(0, oldText)
        widget.focus_set()

    return isValid


def numericInRangeValidator(widget, oldText, newText, start, end):
    """
    A function used to validate the input of an entry, to make sure it is numeric
    :param widget: Sender widget
    :param oldText: Input text before the change
    :param newText: Input text
    :param start: Start of range
    :param end: End of range
    :return: Whether text is a valid number or not
    """
    isValid = numericValidator(widget, oldText, newText)

    if isValid:
        isValid = False
        if isNumeric(newText):
            if start <= float(newText) <= end:
                isValid = True
        elif newText == '':
            isValid = True

        if not isValid:
            messagebox.showerror('Illegal Input',
                                 'Input is out of range [{}, {}]. Was: {}'
                                 .format(start, end, newText))
            widget.delete(0, tk.END)
            widget.insert(0, oldText)
            widget.focus_set()

    return isValid


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent):
        tk.Toplevel.__init__(self, parent)

        # Hide the dialog in the task bar. Let it be an inner window of the parent window
        self.transient(parent)

        self.title('App Settings')
        self.iconbitmap(os.path.abspath(os.path.join('resource', 'settings-icon.ico')))
        self.config(background=ctl.BACKGROUND_COLOR)
        self.parent = parent

        # The result of the dialog: settings.Settings object
        self.__result = None
        self.closing = False  # Marker to see if we cancel the window, to avoid of validating input
        self.gammaCorrectionValueSpinbox = None  # tk.Spinbox
        self.blurKernelSizeSpinbox = None  # tk.Spinbox
        self.gradientEdgeCheckVar = None  # tk.IntVar - to hold the value of the gradient checkbox
        self.gradientEdgeCheckButton = None  # tk.Checkbutton
        self.threshold1Spinbox = None  # tk.Spinbox
        self.threshold2Spinbox = None  # tk.Spinbox
        self.brightBackgroundCheckVar = None  # tk.IntVar - to hold the value of the bright checkbox
        self.brightBackgroundCheckButton = None  # tk.Checkbutton
        self.morphCloseIterationsCountSpinbox = None  # tk.Spinbox
        self.morphOpenIterationsCountSpinbox = None  # tk.Spinbox
        self.morphDilateIterationsCountSpinbox = None  # tk.Spinbox
        self.morphErodeIterationsCountSpinbox = None  # tk.Spinbox
        self.structuringElementDontCareWidthSpinbox = None  # tk.Spinbox
        self.imageShape = settingsInstance.imageShape  # Tuple (Width, Height)
        self.imageShapeEntry = None  # tk.Entry
        self.markThicknessSpinbox = None  # tk.Spinbox
        self.markColor = settingsInstance.markColor  # Tuple (R, G, B)
        self.markColorEntry = None  # tk.Entry
        self.morphologicalMaskShape = settingsInstance.morphologicalMaskShape  # Tuple (Width, Height)
        self.morphologicalMaskShapeEntry = None  # tk.Entry
        self.objectRotationDegreeIncSpinbox = None  # tk.Spinbox

        # Build the body
        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        # Dialog buttons
        self.buttonbox()

        # Make the dialog modal
        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        # Make sure an explicit close is treated as a CANCEL
        self.protocol("WM_DELETE_WINDOW", self.cancel)

        ctl.center(self)
        self.initial_focus.focus_set()
        self.wait_window(self)

    @property
    def result(self):
        """
        The result of settings dialog is None in case user cancelled the dialog, or
        util.settings.Settings object if user approved the dialog
        :return: The settings
        """
        return self.__result

    def body(self, master):
        """
        Dialog body - a frame containing all setting controls
        :param master: Master window
        :return: The created frame
        """
        master.columnconfigure(0, weight=1)
        frame = tk.Frame(master, bg=ctl.BACKGROUND_COLOR)
        frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        frame.columnconfigure(1, weight=1)

        # Row number in grid, to set components row by row
        r = 0
        self.initGammaCorrectionEditor(frame, r)

        r += 1
        self.initBlurKernelSizeEditor(frame, r)

        r += 1
        self.gradientEdgeCheckButton, self.gradientEdgeCheckVar = \
            ctl.checkButton(frame, 'Use Gradient Edge Detector')
        self.gradientEdgeCheckButton.grid(row=r, columnspan=2, padx=5, pady=5, sticky=tk.W)

        r += 1
        self.initThreshold1Editor(frame, r)

        r += 1
        self.initThreshold2Editor(frame, r)

        r += 1
        self.brightBackgroundCheckButton, self.brightBackgroundCheckVar = \
            ctl.checkButton(frame, 'Bright Background Images')
        self.brightBackgroundCheckButton.grid(row=r, columnspan=2, padx=5, pady=5, sticky=tk.W)

        r += 1
        self.morphCloseIterationsCountSpinbox = \
            self.createMorphIterationsCountEditor(frame,
                                                  r,
                                                  'Iterations of Morphological Closing',
                                                  lambda event: self.closeIterationsCountValidator(
                                                      settingsInstance.morphCloseIterationsCount,
                                                      self.morphCloseIterationsCountSpinbox.get()),
                                                  self.closeIterationsCountValidator)

        r += 1
        self.morphOpenIterationsCountSpinbox = \
            self.createMorphIterationsCountEditor(frame,
                                                  r,
                                                  'Iterations of Morphological Opening',
                                                  lambda event: self.openIterationsCountValidator(
                                                      settingsInstance.morphOpenIterationsCount,
                                                      self.morphOpenIterationsCountSpinbox.get()),
                                                  self.openIterationsCountValidator)

        r += 1
        self.morphDilateIterationsCountSpinbox = \
            self.createMorphIterationsCountEditor(frame,
                                                  r,
                                                  'Iterations of Morphological Dilation (Search)',
                                                  lambda event: self.dilateIterationsCountValidator(
                                                      settingsInstance.morphDilateIterationsCount,
                                                      self.morphDilateIterationsCountSpinbox.get()),
                                                  self.dilateIterationsCountValidator)

        r += 1
        self.morphErodeIterationsCountSpinbox = \
            self.createMorphIterationsCountEditor(frame,
                                                  r,
                                                  'Iterations of Morphological Erosion (Search)',
                                                  lambda event: self.erodeIterationsCountValidator(
                                                      settingsInstance.morphErodeIterationsCount,
                                                      self.morphErodeIterationsCountSpinbox.get()),
                                                  self.erodeIterationsCountValidator)

        r += 1
        self.structuringElementDontCareWidthSpinbox = \
            self.createMorphIterationsCountEditor(frame,
                                                  r,
                                                  'Border Width of Don\'tCare in Structuring Element',
                                                  lambda event: self.structuringElementDontCareValidator(
                                                      settingsInstance.structuringElementDontCareWidth,
                                                      self.structuringElementDontCareWidthSpinbox.get()),
                                                  self.structuringElementDontCareValidator)

        r += 1
        self.initMarkColorEditor(frame, r)

        r += 1
        self.initMarkThicknessEditor(frame, r)

        r += 1
        self.initImageShapeEditor(frame, r)

        r += 1
        self.initMorphologicalMaskShapeEditor(frame, r)

        r += 1
        self.initObjectRotationDegreeInc(frame, r)

        # Load settings object to the editors
        self.initSettings()

        return frame

    def initGammaCorrectionEditor(self, frame, r):
        ctl.label(frame, text='Gamma Correction Value').grid(row=r, padx=5, pady=5, sticky=tk.W)
        # np.arange yields bad values.. e.g. -1.60000000e00 or 1.599
        self.gammaCorrectionValueSpinbox = \
            ctl.spinBox(frame,
                        [-1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0,
                         -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                        3,
                        (self.master.register(self.gammaCorrectionValidator),
                         '%s',
                         '%P'))
        self.gammaCorrectionValueSpinbox.bind(
            '<FocusOut>',
            lambda event: self.gammaCorrectionValidator(settingsInstance.gammaCorrectionValue,
                                                        self.gammaCorrectionValueSpinbox.get()))
        self.gammaCorrectionValueSpinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initBlurKernelSizeEditor(self, frame, r):
        ctl.label(frame, text='Blur Kernel Size').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.blurKernelSizeSpinbox = \
            ctl.spinBox(frame,
                        np.arange(start=1, stop=40, step=2).tolist(),
                        3,
                        (self.master.register(self.blurKernelSizeValidator),
                         '%s',
                         '%P'))
        self.blurKernelSizeSpinbox.bind(
            '<FocusOut>',
            lambda event: self.blurKernelSizeValidator(settingsInstance.blurKernelSize,
                                                       self.blurKernelSizeSpinbox.get()))
        self.blurKernelSizeSpinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initThreshold1Editor(self, frame, r):
        ctl.label(frame, text='Threshold1 (Min)').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.threshold1Spinbox = \
            ctl.spinBox(frame,
                        np.arange(start=0, stop=255, dtype=np.uint8).tolist(),
                        3,
                        (self.master.register(self.pixelValueThreshold1Validator),
                         '%s',
                         '%P'))
        self.threshold1Spinbox.bind(
            '<FocusOut>',
            lambda event: self.pixelValueThreshold1Validator(settingsInstance.threshold1,
                                                             self.threshold1Spinbox.get()))
        self.threshold1Spinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initThreshold2Editor(self, frame, r):
        ctl.label(frame, text='Threshold2 (Max)').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.threshold2Spinbox = \
            ctl.spinBox(frame,
                        np.arange(start=1, stop=256, dtype=np.uint8).tolist(),
                        3,
                        (self.master.register(self.pixelValueThreshold2Validator),
                         '%s',
                         '%P'))
        self.threshold2Spinbox.bind(
            '<FocusOut>',
            lambda event: self.pixelValueThreshold2Validator(settingsInstance.threshold2,
                                                             self.threshold2Spinbox.get()))
        self.threshold2Spinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def createMorphIterationsCountEditor(self, frame, r, text, validatorFunc, validatorFuncRef):
        ctl.label(frame, text=text).grid(row=r, padx=5, pady=5, sticky=tk.W)
        # Registering validation command
        morphIterationsCountSpinbox = \
            ctl.spinBox(frame,
                        np.arange(start=1, stop=21, dtype=np.uint8).tolist(),
                        3,
                        (self.master.register(validatorFuncRef), '%s', '%P'))
        morphIterationsCountSpinbox.bind('<FocusOut>', validatorFunc)
        morphIterationsCountSpinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

        return morphIterationsCountSpinbox

    def initMarkColorEditor(self, frame, r):
        ctl.label(frame, text='Highlight Color').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.markColorEntry = tk.Entry(frame,
                                       font=ctl.FONT_REGULAR_BOLD,
                                       foreground='black',
                                       background=ctl.colorToHex(settingsInstance.markColor))
        self.markColorEntry.bind('<FocusOut>',
                                 lambda event: self.colorValidatorCmd(self.markColorEntry.get()))
        self.markColorEntry.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initMarkThicknessEditor(self, frame, r):
        ctl.label(frame, text='Highlight Thickness').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.markThicknessSpinbox = \
            ctl.spinBox(frame,
                        np.arange(start=1, stop=11).tolist(),
                        3,
                        (self.master.register(self.markThicknessValidator),
                         '%s',
                         '%P'))
        self.markThicknessSpinbox.bind(
            '<FocusOut>',
            lambda event: self.markThicknessValidator(settingsInstance.threshold2,
                                                      self.markThicknessSpinbox.get()))
        self.markThicknessSpinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initImageShapeEditor(self, frame, r):
        ctl.label(frame, text='Image Shape').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.imageShapeEntry = ctl.entry(frame)
        self.imageShapeEntry.bind('<FocusOut>',
                                  lambda event: self.shapeValidatorCmd(self.imageShapeEntry.get()))
        self.imageShapeEntry.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initMorphologicalMaskShapeEditor(self, frame, r):
        ctl.label(frame, text='Morphological Mask Shape').grid(row=r, padx=5, pady=5, sticky=tk.W)
        self.morphologicalMaskShapeEntry = ctl.entry(frame)
        self.morphologicalMaskShapeEntry.bind(
            '<FocusOut>',
            lambda event: self.maskShapeValidatorCmd(self.morphologicalMaskShapeEntry.get()))
        self.morphologicalMaskShapeEntry.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def initObjectRotationDegreeInc(self, frame, r):
        ctl.label(frame, text="Object Rotation Degree Increase").grid(row=r, padx=5, pady=5, sticky=tk.W)
        # Registering validation command
        self.objectRotationDegreeIncSpinbox = \
            ctl.spinBox(frame,
                        np.arange(start=1, stop=91, dtype=np.uint8).tolist(),
                        3,
                        (self.master.register(self.objectRotationDegreeValidator), '%s', '%P'))
        self.objectRotationDegreeIncSpinbox.bind(
            '<FocusOut>',
            lambda event: self.objectRotationDegreeValidator(settingsInstance.objectRotationDegreeInc,
                                                             self.objectRotationDegreeIncSpinbox.get()))
        self.objectRotationDegreeIncSpinbox.grid(row=r, column=1, padx=5, pady=5, sticky=tk.EW)

    def buttonbox(self):
        """
        OK and Cancel buttons
        :return: A frame containing the two buttons
        """
        box = ctl.frame(self, tk.BOTH, 0, 0)

        w = tk.Button(box,
                      text="Reset",
                      width=5,
                      height=1,
                      command=self.reset,
                      font=ctl.FONT_REGULAR,
                      foreground='white',
                      background=ctl.BACKGROUND_TOOLTIP_COLOR)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box,
                      text="Cancel",
                      width=10,
                      command=self.cancel,
                      font=ctl.FONT_BUTTON,
                      foreground='white',
                      background='#C75450')
        w.pack(side=tk.RIGHT, padx=5, pady=5)
        w = tk.Button(box,
                      text="OK",
                      width=10,
                      command=self.ok,
                      default=tk.ACTIVE,
                      font=ctl.FONT_BUTTON,
                      foreground='white',
                      background=ctl.ACCEPT_COLOR)
        w.pack(side=tk.RIGHT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def ok(self, event=None):
        """
        Command of the OK button
        :param event:
        :return: None
        """
        error_message = self.validate()
        if error_message is not None:
            messagebox.showerror('Illegal Input', error_message)

            # Put focus back
            self.initial_focus.focus_set()
            return

        # Validate the color specifically because we have not registered a validation
        # command on it, but a FocusOut binding only. The reason is we do want to allow
        # several edits of the color as it is a Tuple string, and validation command
        # validates each single input char..
        if not self.colorValidatorCmd(self.markColorEntry.get()):
            self.initial_focus.focus_set()

        self.withdraw()
        self.update_idletasks()

        self.apply()
        self.cancel()

    def cancel(self, event=None):
        """
        Command of the CANCEL button
        :param event:
        :return: None
        """
        # Sign that we are closing the window, so we will not perform any validation during exit
        self.closing = True

        # Put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    def validate(self):
        """
        Validates the input
        :return: None if valid, error message otherwise
        """
        error = None

        thresh1 = self.threshold1Spinbox.get()
        thresh2 = self.threshold2Spinbox.get()
        if not isNumeric(thresh1) or not isNumeric(thresh2):
            error = 'Thresholds must be numbers'
        elif int(thresh2) <= int(thresh1):
            error = 'Threshold1 must be less than threshold2'

        return error

    def apply(self):
        """
        Gather data into settings variable and set it as the result
        :return: None
        """
        settingsInstance.gammaCorrectionValue = float(self.gammaCorrectionValueSpinbox.get())
        settingsInstance.blurKernelSize = int(self.blurKernelSizeSpinbox.get())
        settingsInstance.isUsingGradientEdgeDetector = bool(self.gradientEdgeCheckVar.get())
        settingsInstance.threshold1 = int(self.threshold1Spinbox.get())
        settingsInstance.threshold2 = int(self.threshold2Spinbox.get())
        settingsInstance.isBrightBackground = bool(self.brightBackgroundCheckVar.get())
        settingsInstance.morphCloseIterationsCount = int(self.morphCloseIterationsCountSpinbox.get())
        settingsInstance.morphOpenIterationsCount = int(self.morphOpenIterationsCountSpinbox.get())
        settingsInstance.morphDilateIterationsCount = int(self.morphDilateIterationsCountSpinbox.get())
        settingsInstance.morphErodeIterationsCount = int(self.morphErodeIterationsCountSpinbox.get())
        settingsInstance.structuringElementDontCareWidth = \
            int(self.structuringElementDontCareWidthSpinbox.get())
        settingsInstance.markColor = self.markColor
        settingsInstance.markThickness = int(self.markThicknessSpinbox.get())
        settingsInstance.imageShape = self.imageShape
        settingsInstance.morphologicalMaskShape = self.morphologicalMaskShape
        settingsInstance.objectRotationDegreeInc = int(self.objectRotationDegreeIncSpinbox.get())
        self.__result = settingsInstance

    def markThicknessValidator(self, oldText, newText):
        """
        A function used to validate the input of iterations count spinbox.
        :param oldText: Text before change
        :param newText: Input text (to the spinbox)
        :return: Whether text is a valid integer (in range) or not
        """
        if self.closing:
            return True

        return numericInRangeValidator(self.markThicknessSpinbox, oldText, newText, 1, 20)

    def closeIterationsCountValidator(self, oldText, newText):
        if self.closing:
            return True
        return numericInRangeValidator(self.morphCloseIterationsCountSpinbox, oldText, newText, 1, 20)

    def openIterationsCountValidator(self, oldText, newText):
        if self.closing:
            return True
        return numericInRangeValidator(self.morphOpenIterationsCountSpinbox, oldText, newText, 1, 20)

    def dilateIterationsCountValidator(self, oldText, newText):
        if self.closing:
            return True
        return numericInRangeValidator(self.morphDilateIterationsCountSpinbox, oldText, newText, 1, 20)

    def erodeIterationsCountValidator(self, oldText, newText):
        if self.closing:
            return True
        return numericInRangeValidator(self.morphErodeIterationsCountSpinbox, oldText, newText, 1, 20)

    def structuringElementDontCareValidator(self, oldText, newText):
        if self.closing:
            return True
        return numericInRangeValidator(self.structuringElementDontCareWidthSpinbox, oldText, newText, 1, 15)

    def gammaCorrectionValidator(self, oldText, newText):
        """
        A function used to validate the input of gamma correction value.
        :param oldText: Text before change
        :param newText: Input text (to the spinbox)
        :return: Whether text is a valid number (in range) or not
        """
        if self.closing:
            return True

        return numericInRangeValidator(self.gammaCorrectionValueSpinbox, oldText, newText, -2.5, 2.5)

    def blurKernelSizeValidator(self, oldText, newText):
        """
        A function used to validate the input of blur kernel size (blurring rate).
        :param oldText: Text before change
        :param newText: Input text (to the spinbox)
        :return: Whether text is a valid integer (in range) or not
        """
        if self.closing or newText == '':
            return True

        if numericInRangeValidator(self.blurKernelSizeSpinbox, oldText, newText, 1, 40):
            if int(newText) % 2 != 1:
                messagebox.showerror('Illegal Input', 'Kernel size must be odd')
                self.blurKernelSizeSpinbox.delete(0, tk.END)
                self.blurKernelSizeSpinbox.insert(0, oldText)
                self.blurKernelSizeSpinbox.focus_set()
            else:
                return True

        return False

    def objectRotationDegreeValidator(self, oldText, newText):
        if self.closing or newText == '':
            return True

        return numericInRangeValidator(self.objectRotationDegreeIncSpinbox, oldText, newText, 1, 91)

    def pixelValueThreshold1Validator(self, oldText, newText):
        """
        A function used to validate the input of thresholds. (0-255)
        :param oldText: Text before change
        :param newText: Input text (to the spinbox)
        :return: Whether text is a valid integer (in range) or not
        """
        if self.closing:
            return True

        return numericInRangeValidator(self.threshold1Spinbox, oldText, newText, 0, 255)

    def pixelValueThreshold2Validator(self, oldText, newText):
        """
        A function used to validate the input of thresholds. (0-255)
        :param oldText: Text before change
        :param newText: Input text (to the spinbox)
        :return: Whether text is a valid integer (in range) or not
        """
        if self.closing:
            return True

        return numericInRangeValidator(self.threshold2Spinbox, oldText, newText, 0, 255)

    def numericValidatorCmd(self, widget_name, oldText, newText):
        """
        A function used to validate the input of an entry, to make sure it is numeric
        :param widget_name: Sender widget
        :param oldText: Input text before the change
        :param newText: Input text
        :return: Whether text is a valid number or not
        """
        if self.closing:
            return True

        return numericValidator(self.nametowidget(widget_name), oldText, newText)

    def colorValidatorCmd(self, text):
        if self.closing:
            return True

        isValid = False
        text = text.strip()
        if isTuple(text):
            isValid = True
            for val in literal_eval(text):
                if val < 0 or val > 255:
                    isValid = False
                    break
        elif text == '':
            isValid = True

        if not isValid:
            messagebox.showerror('Illegal Input', 'Input color is illegal. Was: {}'.format(text))
            self.markColorEntry.focus_set()
        else:
            self.markColor = literal_eval(text)
            self.markColorEntry.configure(background=ctl.colorToHex(self.markColor))

        return isValid

    def shapeValidatorCmd(self, text):
        if self.closing:
            return True

        isValid = False
        text = text.strip()
        if isTuple(text):
            tupleElements = literal_eval(text)
            if tupleElements.length == 2:
                isValid = True
                for val in tupleElements:
                    if val < 1:
                        isValid = False
                        break
        elif text == '':
            isValid = True

        if not isValid:
            messagebox.showerror('Illegal Input', 'Image shape is illegal. Was: {}'.format(text))
            self.imageShapeEntry.focus_set()
        else:
            self.imageShape = literal_eval(text)

        return isValid

    def maskShapeValidatorCmd(self, text):
        if self.closing:
            return True

        isValid = False
        text = text.strip()
        if isTuple(text):
            tupleElements = literal_eval(text)
            if tupleElements.length != 2:
                isValid = False
            else:
                isValid = True
                for val in tupleElements:
                    if val < 1 or val > 99:
                        isValid = False
                        break
        elif text == '':
            isValid = True

        if not isValid:
            messagebox.showerror('Illegal Input', 'Mask shape is illegal. Was: {}'.format(text))
            self.morphologicalMaskShapeEntry.focus_set()
        else:
            self.morphologicalMaskShape = literal_eval(text)

        return isValid

    def reset(self):
        """
        Resets settings to factory settings (defaults)
        :return: None
        """
        settingsInstance.reset()
        self.initSettings()

    def initSettings(self):
        """
        Load settings from settings object into the editors
        :return: None
        """
        self.gradientEdgeCheckVar.se = settingsInstance.isUsingGradientEdgeDetector
        if settingsInstance.isUsingGradientEdgeDetector:
            self.gradientEdgeCheckButton.select()
        else:
            self.gradientEdgeCheckButton.deselect()

        if settingsInstance.isBrightBackground:
            self.brightBackgroundCheckButton.select()
        else:
            self.brightBackgroundCheckButton.deselect()

        self.markColor = settingsInstance.markColor
        self.markColorEntry.configure(background=ctl.colorToHex(self.markColor))
        resetText(self.markColorEntry, settingsInstance.markColor)
        resetText(self.gammaCorrectionValueSpinbox, float(settingsInstance.gammaCorrectionValue))
        resetText(self.blurKernelSizeSpinbox, int(settingsInstance.blurKernelSize))
        resetText(self.threshold1Spinbox, int(settingsInstance.threshold1))
        resetText(self.threshold2Spinbox, int(settingsInstance.threshold2))
        resetText(self.morphCloseIterationsCountSpinbox, int(settingsInstance.morphCloseIterationsCount))
        resetText(self.morphOpenIterationsCountSpinbox, int(settingsInstance.morphOpenIterationsCount))
        resetText(self.morphDilateIterationsCountSpinbox, int(settingsInstance.morphDilateIterationsCount))
        resetText(self.morphErodeIterationsCountSpinbox, int(settingsInstance.morphErodeIterationsCount))
        resetText(self.structuringElementDontCareWidthSpinbox,
                  int(settingsInstance.structuringElementDontCareWidth))
        resetText(self.markThicknessSpinbox, int(settingsInstance.markThickness))
        self.imageShape = settingsInstance.imageShape
        resetText(self.imageShapeEntry, settingsInstance.imageShape)
        self.morphologicalMaskShape = settingsInstance.morphologicalMaskShape
        resetText(self.morphologicalMaskShapeEntry, settingsInstance.morphologicalMaskShape)
        resetText(self.objectRotationDegreeIncSpinbox, int(settingsInstance.objectRotationDegreeInc))
