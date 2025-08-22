# giga_controller_local.py
from __future__ import annotations
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
import clipboard
from loguru import logger
from pywinauto.timings import wait_until_passes, TimeoutError
from pywinauto import timings
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
from pywinauto.findwindows import ElementNotFoundError
from pywinauto.timings import TimeoutError
from pywinauto import Desktop
from pywinauto import mouse
try:
    import win32api, win32con
except Exception:
    win32api = None
    win32con = None
import re
from PIL import ImageGrab, Image
import subprocess
import time
from playwright.sync_api import sync_playwright

import os
from Pywinauto import Pywinauto

class ModeType(Enum):
    SCALE = 'Scale'
    WIDTH = 'Width'
    HEIGHT = 'Height'

class Mode(Enum):
    X05 = "0.5x"
    X2 = "2x"
    X4 = "4x"
    X6 = "6x"

class Dimension(Enum):
    PX = "px"
    IN = "in"
    CM = "cm"

class Model(Enum):
    STANDARD = "Standard"
    ART_AND_CG = "Art & CG"
    LOW_RESOLUTION = "Low Resolution"
    LINES = "Lines"
    VERY_COMPRESSED = "Very Compressed"
    HIGH_FIDELITY = "High fidelity"  # may not exist as button in dump


class SaveFormat(Enum):
    SAME = "Preserve source format"
    PNG = "PNG"
    JPG = "JPG"
    JPEG = "JPEG"
    TIF = "TIF"
    TIFF = "TIFF"
    DNG = "DNG"

class GigaPixel(Pywinauto):
    """
    Controller tailored to the Topaz Gigapixel UI you provided.
    """
    def __init__(self, executable_path: Union[str, Path], backend: str = "uia", window_find_timeout: float = 2, after_click_wait: float = 0.5):
        super().__init__(executable_path=executable_path, window_find_timeout=window_find_timeout, after_click_wait=after_click_wait)
        
        self.backend = backend


    def openApp(self, start_if_missing: bool = True) -> None:
        """Connect to an existing Gigapixel process or start a new one."""
        try:
            self.app = Application(backend=self.backend).connect(path=str(self.exe))
            logger.debug("Connected to existing Gigapixel process.")
        except Exception:
            if not start_if_missing:
                raise
            logger.debug("Starting new Gigapixel instance...")
            self.app = Application(backend=self.backend).start(str(self.exe))
        
        # Cache main window (matches your dump "Topaz Gigapixel AI")
        self.main_win = self.app.window(title_re=".*Gigapixel.*")
        self.main_win.wait('visible')
        self.main_win.set_focus()
        #self.main_win.print_control_identifiers()
        
    def closeApp(self):
        try:
            if not self.app:
                return
            try:
                self.closeImages()
                self.main_win.close()
            except Exception:
                self.app.kill()
        finally:
            self.app = None
            self.main_win = None
            
    def openImage(self, path):
        path = Path(path)
        if not path.exists():
            raise ValueError(f'Image path do not exist: {path.as_posix()}')
        
        self.main_win.set_focus()
        send_keys('^o')
        clipboard.copy(path.as_posix())
        
        # Get the dialogue box
        open_dlg = Desktop(backend='win32').window(title_re=".*Open.*")
        open_dlg.wait('visible')
        
        # Go to folder
        self.fileSysWinGoToPath(path = path.parent.as_posix())

        # Enter filename directly into File name box 
        open_dlg['File name:Edit'].set_edit_text(path.name)

        # Click Open button explicitly 
        open_dlg['Open'].click()
        
        # self.main_win.print_control_identifiers()
    
    def openImages(self):
        raise NotImplementedError
        
    def closeImage(self):
        raise NotImplementedError

    def closeImages(self, save=True):
        self.main_win.set_focus()
        send_keys('^w')
        self.imageNotSavePopUp(save=save)
        send_keys('^w')
        
    def saveImage(self, filename:str='', save_dir:str=''):
        def findName(control_type):
            for el_dict in cis_differences['added']:
                if el_dict['control_type'] == control_type:
                    return el_dict['name']
            print(cis_differences)
            raise Exception(f'The {control_type} name as not been found')
        self.removePopUp()
        cis_before = self.getControlIdentifiers(self.main_win)
        send_keys('^s')
        cis_after = self.getControlIdentifiers(self.main_win)
        cis_differences = self.differenceControlIdentifiers(cis_before, cis_after)
        
        edit_filename_name = findName('Edit')
        combo_box_format_name = findName('ComboBox')
        
        self.imageSavePopUp(combo_box_format_name=combo_box_format_name, edit_filename_name=edit_filename_name, filename=filename, save_dir=save_dir)
        
    def imageSavePopUp(self, combo_box_format_name:str, edit_filename_name:str, filename:str='', save_dir:str=''):
        save_btn = self.main_win.child_window(title='Save', control_type='Button') # If one Image
        
        # Todo output file type(png, jpeg to implement.)
        edit_filename = self.main_win.child_window(title=edit_filename_name)
        combo_box_format = self.main_win.child_window(title=combo_box_format_name)
            
        # Get positions for folder
        edit_filename_rec, combo_box_format_rec = edit_filename.rectangle(), combo_box_format.rectangle()
        
        margin = edit_filename_rec.top - combo_box_format_rec.bottom
        height = edit_filename_rec.bottom - edit_filename_rec.top
        y = int(edit_filename_rec.bottom + margin + height // 2)
        button_size = (edit_filename_rec.right - edit_filename_rec.left) / 2
        positions = {btn_name: int(edit_filename_rec.left + (i + 0.5) * button_size) for i, btn_name in enumerate(['Source', 'Custom'])}

        if len(save_dir) > 0:

            mouse.click(button='left', coords=(positions['Custom'], y))
            btn_change = self.main_win.child_window(title='Change').click_input()
            
            open_dlg = Desktop(backend='win32').window(title_re=".*Output Folder.*")
            open_dlg.wait('visible')
            
            self.fileSysWinGoToPath(save_dir)
            open_dlg['Select Folder'].click()
        else:
            mouse.click(button='left', coords=(positions['Source'], y))
            
        if len(filename) > 0:
            edit_filename.set_edit_text(filename)
            
        save_btn.click_input()
        self.waitUntilDisappears(self.main_win.child_window(title='Cancel Processing', control_type='Button'))
        
        return True
        
    def saveImages(self):
        """
        Form is different from 1 to many images
        """
        raise NotImplementedError()
        self.imagesSavePopUp(self)
        
    def imagesSavePopUp(self):
        raise NotImplementedError()
        save_btn = self.main_win.child_window(title='Start', control_type='Button') # If many images
    
        
    def imageNotSavePopUp(self, save=False):
        save_btn = self.main_win.child_window(title="Save", control_type="Button")
        close_btn = self.main_win.child_window(title="Close Without Saving", control_type="Button")
        cancel_btn = self.main_win.child_window(title="Cancel", control_type="Button")
        
        if not close_btn.exists():
            logger.debug("Close Without Saving do NOT exist")
            return False
        
        if save:
            save_btn.click_input()
            self.imageSavePopUp()
        else:
            close_btn.click_input()
        return True
        
    def selectModeType(self, mode_type:ModeType = ModeType.SCALE):
        mode_panel = self.main_win.child_window(title='Resize Mode', control_type='Button')
        
        # Get element above and bellow for target the y and then the x positions
        top_element = self.main_win.child_window(title='Crop', control_type='Button')
        bottom_elements = [
                            self.main_win.child_window(title=Mode.X2.value),
                            *[self.main_win.child_window(title=dim.value, control_type='ComboBox') for dim in Dimension]
                        ]
        for el in bottom_elements:
            if el.exists():
                bottom_element = el
                break
        else:
            print(self.main_win.print_control_identifiers())
            raise Exception('No valid bottom element foud')
        
        # Get positions
        top_rec, bottom_rec, mode_panel_rec = top_element.rectangle(), bottom_element.rectangle(), mode_panel.rectangle()

        # Define y position
        y = (top_rec.bottom + bottom_rec.top) // 2
        
        # Define x positions
        mode_type_size =  (top_rec.right - mode_panel_rec.left) / len(ModeType)
        xs = {cur_mode_type.value: int(mode_panel_rec.left + (i + 0.5) * mode_type_size) for i, cur_mode_type in enumerate(ModeType)}

        # Select the desire position
        mouse.click(button='left', coords=(xs[mode_type.value], y))
        
    def selectMode(self, mode:Mode = Mode.X2):
        element = self.main_win.child_window(title=mode.value).click_input()
        
    def fillSize(self, size):
        # todo improve for allow other than px (in, cm)
        for pane in self.main_win.descendants(control_type="Pane"):
            px = None
            for child in pane.children():
                if child.window_text() == "px":
                    px = child
                    break

            if not px:
                continue 
            
            pane_childrens = pane.children()
            px_idx = pane_childrens.index(px)
            edit = pane_childrens[px_idx - 1]

            if edit.friendly_class_name() != 'Edit':
                raise AttributeError("Error when tried to get the edit. This method is wrong, modify its behavior.")

            edit.set_edit_text(str(size))
            return True

        raise Exception("Failed to find the px dropbox")
    
    def upScaleImages(self):
        raise NotImplementedError()
        
    def upScaleImage(self, mode_type:ModeType = ModeType.SCALE, mode:Mode|int = Mode.X2):
        if mode_type == ModeType.SCALE:
            if not isinstance(mode, Mode):
                raise AttributeError(f'With ModeType.SCALE, the mode should be an instance of Mode.')
            
            self.selectModeType(mode_type)
            self.selectMode(mode=mode)
        elif mode_type != ModeType.SCALE:
            if isinstance(mode, Mode):
                raise ArithmeticError(f'With  ModeType.{mode_type.name}, the mode should be an instance of int.')
            
            self.selectModeType(mode_type)
            self.fillSize(size=mode)
        
        self.saveImage()
        
    def upScaleImageWithMaximalSize(self):
        raise NotImplementedError()
        
    def upScaleImageWithMinimalSize(self,image_path,  min_width, min_height):
        self.closeImages(save=False)
        self.openImage(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
        
        width_factor, height_factor = min_width / width, min_height / height
        
        if width_factor < 1 and height_factor < 1:
            raise Exception('The min sizes provided do not allow an upscale:\n- Width factor: {round(width_factor, 2)}\n- Height factor: {round(height_factor, 2)}')
        
        print(width_factor, width, min_width)
        print(height_factor, height, min_height)
        
        if width_factor > height_factor:
            self.upScaleImage(mode_type=ModeType.WIDTH, mode=min_width)
        else:
            self.upScaleImage(mode_type=ModeType.HEIGHT, mode=min_height)
            
    def removePopUp(self):
        send_keys('^,')
        send_keys('{ESC}')
        
        
        
if __name__ == '__main__':
    gp = GigaPixel(r"C:\Program Files\Topaz Labs LLC\Topaz Gigapixel AI\Topaz Gigapixel AI.exe")
    gp.openApp()
    gp.openImage(path=r'C:\Users\esto5\Downloads\original.png')
    gp.saveImage('testtest',r'C:\Users\esto5\Downloads\save')
    