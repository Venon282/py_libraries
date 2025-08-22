import re
import clipboard
from pathlib import Path
from pywinauto import Desktop
from pywinauto import timings
from typing import Optional, Union
from pywinauto.keyboard import send_keys
from pywinauto.application import Application
from pywinauto.timings import wait_until_passes
from pywinauto.findwindows import ElementNotFoundError

class Pywinauto:
    def __init__(self, executable_path: Union[str, Path], window_find_timeout: float = 2, after_click_wait: float = 0.5):
        self.exe = Path(executable_path)

        self.app: Optional[Application] = None
        self.main_win = None

        # tune find timeout
        timings.Timings.window_find_timeout = window_find_timeout
        timings.after_click_wait = after_click_wait
        
    def openApp(self, start_if_missing: bool = True) -> None:
        """Connect to an existing Gigapixel process or start a new one."""
        try:
            self.app = Application(backend=self.backend).connect(path=str(self.exe))
        except Exception:
            if not start_if_missing:
                raise
            self.app = Application(backend=self.backend).start(str(self.exe))

    def closeApp(self):
        try:
            if not self.app:
                return
            try:
                self.main_win.close()
            except Exception:
                self.app.kill()
        finally:
            self.app = None
            self.main_win = None
            
    def screenShotElements(self, output_dir = "ui_element_screenshots", parents=True, exist_ok=True, verbose=0):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=parents, exist_ok=exist_ok)

        # Get all descendant elements
        elements =self.main_win.descendants()
        self.main_win.set_focus()

        # Iterate and screenshot
        for i, elem in enumerate(elements):
            try:
                ctrl_type = elem.friendly_class_name()
                title = elem.window_text() or "NoTitle"
                rect = elem.rectangle()

                # Clean title for filename
                title_clean = re.sub(r'[^a-zA-Z0-9_]+', '_', title.strip())[:50]

                # Format the name like: Button_Save_2_Images_(L921_T637_R1191_B677)
                filename = f"{ctrl_type}_{title_clean}_(L{rect.left}_T{rect.top}_R{rect.right}_B{rect.bottom}).png"
                filepath =  output_dir / filename
                self.screenShotElement(elem, filepath, focus=False, verbose=verbose)
            except Exception as e:
                print(f"Failed to capture element {i}: {e}")
    
    def screenShotElement(self, element, filename:str='element.png', focus=True, overwrite=False, verbose=0):
        filename = Path(filename)
        
        if not overwrite and filename.exists():
            if verbose:
                print(f"Can\'t save {filename.as_posix()} because it already exist")
            return False
        
        if focus:
            self.main_win.set_focus()
        # rect = element.rectangle()
        # img = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom))
        img = element.capture_as_image()
        img.save(filename)
        
        if verbose:
            print(f'Saved: {filename}')
            
    def displayAvailibleWindows(self, backend='uia'):
        for w in Desktop(backend="uia").windows():
            print(w.window_text())
            
    def waitUntilDisappears(self, element, timeout=60, retry_interval=1):
        def check():
            try:
                if element.exists() and element.is_visible():
                    raise RuntimeError("Button still visible")
            except ElementNotFoundError:
                return True
            return True

        wait_until_passes(timeout, retry_interval, check)
        
    def getControlIdentifiers(self, ctrl):
        return [{
                "control_type": getattr(child.element_info, "control_type", None),
                "class_name": child.friendly_class_name(),
                "automation_id": getattr(child.element_info, "automation_id", None),
                "name": child.window_text(),
                'rectangle': child.rectangle(),
            } for child in ctrl.children()]
            
    def getControlIdentifiersRec(self, ctrl, depth=0, out=None):
        if out is None:
            out = []

        # Add this control’s info
        out.append({
            "depth": depth,
            "control_type": getattr(ctrl.element_info, "control_type", None),
            "class_name": ctrl.friendly_class_name(),
            "automation_id": getattr(ctrl.element_info, "automation_id", None),
            "name": ctrl.window_text(),
            'rectangle': ctrl.rectangle(),
        })

        # Recurse into children
        for child in ctrl.children():
            self.getControlIdentifiers(child, depth + 1, out)

        return out
    
    def differenceControlIdentifiers(self, cis1, cis2):
        """
        Compare two control identifier lists (cis1, cis2) and
        return dict with 'added', 'removed', 'changed'.
        Each cis must be a list of dicts from getControlIdentifiers/getControlIdentifiersRec.
        """

        def sig(c):
            """Unique-ish signature for comparison"""
            return (
                c.get("automation_id"),
                c.get("name"),
                c.get("class_name"),
                c.get("control_type"),
            )

        dict1 = {sig(c): c for c in cis1}
        dict2 = {sig(c): c for c in cis2}

        set1, set2 = set(dict1.keys()), set(dict2.keys())

        added_keys = set2 - set1
        removed_keys = set1 - set2
        common_keys = set1 & set2

        # detect changed controls (same key but different properties)
        changed = []
        for k in common_keys:
            c1, c2 = dict1[k], dict2[k]
            if c1 != c2:  # dict deep comparison
                changed.append({"before": c1, "after": c2})

        return {
            "added": [dict2[k] for k in added_keys],
            "removed": [dict1[k] for k in removed_keys],
            "changed": changed,
        }
        
    def fileSysWinGoToPath(self, path:str):
        send_keys("^l")  # focus path bar
        clipboard.copy(path)
        send_keys("^v{ENTER}")
        
    