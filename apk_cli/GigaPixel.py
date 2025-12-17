# giga_controller_local.py
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Union
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
from pywinauto import Desktop, mouse
from PIL import Image
from .Pywinauto import Pywinauto
from collections import defaultdict
import time
import logging

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
        except Exception:
            if not start_if_missing:
                raise
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
    
    def openImages(self, images_path:list|dict, wait=0.2):
        self.main_win.set_focus()
        
        for root, filenames in self.iterateImages(images_path):
            send_keys('^o')
            time.sleep(wait)
            
            # Get the dialogue box
            open_dlg = Desktop(backend='win32').window(title_re=".*Open.*")
            open_dlg.wait('visible', timeout=20)
            
            # Go to folder
            self.fileSysWinGoToPath(path = root.as_posix())
            time.sleep(wait)
            
            # Enter filenames directly into File name box 
            open_dlg['File name:Edit'].set_edit_text(' '.join(['"'+filename+'"'for filename in filenames]))
            time.sleep(wait)
            
            # Click Open button explicitly 
            open_dlg['Open'].click()
            time.sleep(wait)
        
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

        btn_change = self.main_win.child_window(title='Change')
        
        if len(save_dir) > 0:
            if not btn_change.exists():
                mouse.click(button='left', coords=(positions['Custom'], y))
            
            btn_change = self.main_win.child_window(title='Change').click_input()
            open_dlg = Desktop(backend='win32').window(title_re=".*Output Folder.*")
            open_dlg.wait('visible')
            
            self.fileSysWinGoToPath(save_dir)
            open_dlg['Select Folder'].click()
        else:
            if btn_change.exists():
                mouse.click(button='left', coords=(positions['Source'], y))
            
        if len(filename) > 0:
            edit_filename.set_edit_text(filename)
            
        save_btn.click_input()
        self.waitUntilDisappears(self.main_win.child_window(title='Cancel Processing', control_type='Button'))
        
        return True
        
    def saveImages(self, suffix:str='', prefix:str='', save_dir:str=''):
        """
        Form is different from 1 to many images
        """
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


        # edit_filename_name = findName('Edit')
        combo_box_format_name = findName('ComboBox')
        
        self.imagesSavePopUp(combo_box_format_name=combo_box_format_name, suffix=suffix, prefix=prefix, save_dir=save_dir)
        
    def imagesSavePopUp(self, combo_box_format_name, suffix:str='', prefix:str='', save_dir:str=''):
        save_btn = self.main_win.child_window(title='Start', control_type='Button') # If many images
        
        # Todo output file type(png, jpeg to implement.)
        combo_box_format = self.main_win.child_window(title=combo_box_format_name)
            
        # Get positions for folder
        save_btn_rec, combo_box_format_rec = save_btn.rectangle(), combo_box_format.rectangle()
        
        margin = 2 # impossible to know
        height = combo_box_format_rec.bottom - combo_box_format_rec.top
        total_height = save_btn_rec.top - combo_box_format_rec.bottom
        y_filename = int(combo_box_format_rec.bottom + margin + height // 2)
        y_directory = int(combo_box_format_rec.bottom + margin//2+ total_height // 2 + height // 2)

        button_size = (combo_box_format_rec.right - combo_box_format_rec.left) / 2
        positions  = {btn_name: int(combo_box_format_rec.left + (i + 0.5) * button_size) for i, btn_name in enumerate(['Origin', 'Custom'])}

        btn_change = self.main_win.child_window(title='Change')
        cis_before = self.getControlIdentifiers(self.main_win)
        time.sleep(1)
        mouse.click(button='left', coords=(positions['Origin'], y_filename))
        time.sleep(1)
        cis_after = self.getControlIdentifiers(self.main_win)
        cis_differences = self.differenceControlIdentifiers(cis_before, cis_after)
        
        
        # Good file options click
        input_prefix = None
        input_suffix = None
        if len(cis_differences['added']) >= 2:
            elements = cis_differences['added']
        elif len(cis_differences['removed']) >= 2:
            mouse.click(button='left', coords=(positions['Origin'], y_filename))
            elements = cis_differences['removed']
        else:
            print(cis_before)
            print(cis_after)
            raise Exception(f'Edit inputs not founs. {cis_differences}')
        
        # prefix and suffix
        elements = [el for el in elements if el['control_type'] == 'Edit']              # Isolate the edits
        for input in self.main_win.descendants(control_type='Edit'):      # For each edits
            r1 = input.rectangle()                                                      # Get it's positions
            for ctrl in elements:                                                       # For edits that we want (or not)
                r2 = ctrl['rectangle']                                                  # Get it's positions
                if input.window_text() == ctrl['name'] and self.isSamePosition(r1, r2): # If we want it
                    
                    # Attribuate it to the good variable
                    if not input_prefix:
                        input_prefix = input
                    elif not input_suffix:                         
                        input_suffix = input
                        break
                    else: break # security
            if input_suffix and input_prefix:
                break 
        else:
            raise Exception('Failed to get the suffix and prefix inputs')
        
        input_suffix.set_edit_text(suffix)
        input_prefix.set_edit_text(prefix)
        
        # Directory save
        if len(save_dir) > 0:
            if not btn_change.exists():
                mouse.click(button='left', coords=(positions['Origin'], y_directory))
            
            btn_change = self.main_win.child_window(title='Change').click_input()
            open_dlg = Desktop(backend='win32').window(title_re=".*Output Folder.*")
            open_dlg.wait('visible')
            
            self.fileSysWinGoToPath(save_dir)
            open_dlg['Select Folder'].click()
        else:
            if btn_change.exists():
                mouse.click(button='left', coords=(positions['Origin'], y_directory))
            
        save_btn.click_input()
        self.waitUntilDisappears(self.main_win.child_window(title='Cancel Processing', control_type='Button'), timeout=3600)
        
        return True
    
        
    def imageNotSavePopUp(self, save:bool=False, verbose:int=0):
        save_btn = self.main_win.child_window(title="Save", control_type="Button")
        close_btn = self.main_win.child_window(title="Close Without Saving", control_type="Button")
        cancel_btn = self.main_win.child_window(title="Cancel", control_type="Button")
        
        if not close_btn.exists():
            if verbose:
                print("Close Without Saving do NOT exist")
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
    
    def upScaleImages(self, mode_type:ModeType = ModeType.SCALE, mode:Mode|int = Mode.X2, suffix:str='', prefix:str='', save_dir:str=''):
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
        
        self.saveImages(suffix=suffix, prefix=prefix, save_dir=save_dir)
        
    def upScaleImage(self, mode_type:ModeType = ModeType.SCALE, mode:Mode|int = Mode.X2, filename:str='', save_dir:str=''):
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
        
        self.saveImage(filename, save_dir)
        
    def upScaleImageWithMaximalSize(self):
        raise NotImplementedError()
    
    def upScaleImagesWithMaximalSize(self):
        raise NotImplementedError()
        
    def upScaleImageWithMinimalSize(self,image_path:str,  min_width:int, min_height:int, filename:str='', save_dir:str=''):
        self.closeImages(save=False)
        self.openImage(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
        
        width_factor, height_factor = min_width / width, min_height / height
        
        if width_factor < 1 and height_factor < 1:
            raise Exception('The min sizes provided do not allow an upscale:\n- Width factor: {round(width_factor, 2)}\n- Height factor: {round(height_factor, 2)}')
        
        if width_factor > height_factor:
            self.upScaleImage(mode_type=ModeType.WIDTH, mode=min_width, filename=filename, save_dir=save_dir)
        else:
            self.upScaleImage(mode_type=ModeType.HEIGHT, mode=min_height, filename=filename, save_dir=save_dir)
            
    def iterateImages(self,images_path:list|dict):
        if isinstance(images_path, list):
            new_images_path = defaultdict(list)
            for path in images_path:
                path = Path(path)
                new_images_path[path.parent.as_posix()].append(path.name)
            images_path = new_images_path 
        elif not isinstance(images_path, dict):
            raise AttributeError(f'images_path should be either instance of dict or list but found {type(images_path)}')
        
        for root, filenames in images_path.items():
            root = Path(root) 
            yield root, filenames
        
            
    def upScaleImagesWithMinimalSize(self,images_path:list|dict,  min_width:int, min_height:int, filename_width:str='', filename_height:str='', suffix:str='', prefix:str='', save_dir:str=''):   
        """
        if only one file: filename else sufix and prefix
        """
        images_width, images_height = defaultdict(list), defaultdict(list)
        for root, filenames in self.iterateImages(images_path):
            for filename in filenames:
                image_path = root / filename
                with Image.open(image_path.as_posix()) as img:
                    width, height = img.size
            
                if width >= min_width and height >= min_height:
                    print(f'Image already at the desire size: {width}, {height}: {image_path.as_posix()}')
                    continue
                
                width_factor, height_factor = min_width / width, min_height / height
                if width_factor > height_factor:
                    images_width[root.as_posix()].append(filename)
                else:
                    images_height[root.as_posix()].append(filename)
    
        if len(images_height) > 0:
            logging.info(f'{len(images_height)} images on the height will be process')
            self.closeImages(save=False)
            self.openImages(images_height)
            if len(images_height) > 1:
                self.upScaleImages(mode_type=ModeType.HEIGHT, mode=min_height, suffix=suffix, prefix=prefix, save_dir=save_dir)
            else:
                self.upScaleImage(mode_type=ModeType.HEIGHT, mode=min_height, filename=filename_height, save_dir=save_dir)
            
        if len(images_width) > 0:
            logging.info(f'{len(images_width)} images on the width will be process')
            self.closeImages(save=False)
            self.openImages(images_width)
            if len(images_width) > 1:
                self.upScaleImages(mode_type=ModeType.WIDTH, mode=min_width, suffix=suffix, prefix=prefix, save_dir=save_dir)
            else:
                self.upScaleImage(mode_type=ModeType.WIDTH, mode=min_width, filename=filename_width, save_dir=save_dir)
            
            
        return images_width, images_height

    def removePopUp(self):
        send_keys('^,')
        send_keys('{ESC}')
        
        
        
if __name__ == '__main__':
    user = 'user_name'
    gp = GigaPixel(r"C:\Program Files\Topaz Labs LLC\Topaz Gigapixel AI\Topaz Gigapixel AI.exe")
    gp.openApp()

    gp.openImages([
        r'E:\gasa\get_and_sell_art\images\13ddf22f-fc3d-45e3-bd2b-02865b00b18a\original.jpeg',
        r'E:\gasa\get_and_sell_art\images\13ddf22f-fc3d-45e3-bd2b-02865b00b18a\redbubble.jpeg',
        r'E:\gasa\get_and_sell_art\images\7b908d63-8dc6-4abb-87e5-a577f0fa729c\original.jpeg',
        r'E:\gasa\get_and_sell_art\images\7b908d63-8dc6-4abb-87e5-a577f0fa729c\redbubble.jpeg',
    ])
    # gp.upScaleImagesWithMinimalSize()
    # gp.openImage(path=r'C:\Users\{user}\Downloads\original.png')
    # gp.saveImage('testtest',fr'C:\Users\{user}\Downloads\save')
    