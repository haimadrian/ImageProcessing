__author__ = "Haim Adrian"

import os
import tkinter as tk
import tkinter.ttk as ttk
import view.controls as ctl
from view.tooltip import Tooltip
import tkinter.filedialog as tkfiledialog


class FileInput(tk.Frame):
    def __init__(self, master, tooltip, imageOnly=True):
        tk.Frame.__init__(self, master)

        self.background = ctl.BACKGROUND_COLOR
        self.configure(background=ctl.BACKGROUND_COLOR)

        self.imageOnly = imageOnly
        self.openFileButton = None  # tk.Button
        self.filePathEntry = None  # tk.Entry
        self.openFileButtonTooltip = None  # view.tooltip.Tooltip
        self.filePathEntryTooltip = None  # view.tooltip.Tooltip
        self.magnifyingIcon = None  # tk.PhotoImage

        self.magnifyingIcon = tk.PhotoImage(file=os.path.abspath(
            os.path.join('resource', 'magnifying-icon.png')))
        self.filePathEntry = ctl.entry(self)
        self.filePathEntry.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.filePathEntryTooltip = Tooltip(self.filePathEntry, tooltip)

        self.openFileButton = ttk.Button(master=self,
                                         image=self.magnifyingIcon,
                                         command=self.openFileAction,
                                         style='TButton',
                                         width=4)
        self.openFileButton.pack(side=tk.LEFT)
        self.openFileButtonTooltip = Tooltip(self.openFileButton, tooltip)

    def openFileAction(self):
        """
        Displaying a file open dialog in image selecting mode, to select an image for executing
        the algorithm on
        :return: None
        """
        filetypes = [("Image File", '.jpg .jpeg .png .bmp')]
        if not self.imageOnly:
            filetypes.append(("Video File", '.mp4 .avi .wmv'))

        fileName = tkfiledialog.askopenfilename(filetypes=filetypes)
        self.filePathEntry.delete(0, tk.END)
        self.filePathEntry.insert(0, fileName)

    def getSelectedFilePath(self):
        return self.filePathEntry.get().strip()

    def setState(self, newState):
        self.filePathEntry["state"] = newState
        self.openFileButton["state"] = newState
