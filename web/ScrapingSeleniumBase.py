from seleniumbase import SB
from selenium.webdriver.common.by import By
import random
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    WebDriverException,
    TimeoutException,
)
import pyperclip
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import time
import traceback
import pickle
from pathlib import Path
import pyautogui
import logging

class ScrapingSeleniumBase:
    
    def __init__(self):
        self._started = False
        self._mouse_position = None
        
    def start(self, **kwargs):
        self.sb_context = SB(
            browser         =   kwargs.get('browser'        , 'chrome'),
            uc              =   kwargs.get('uc'             , True),  # undetected chromium
            headless        =   kwargs.get('headless'       , False),
            agent           =   kwargs.get('agent'          , 'random'),
            incognito       =   kwargs.get('incognito'      , True),
            locale_code     =   kwargs.get('locale_code'    , 'en-US'),
            recorder_ext    =   kwargs.get('recorder_ext'   , False),
            disable_csp     =   kwargs.get('disable_csp'    , True),
        ) 
        
        self.sb = self.sb_context.__enter__()
        
        if not kwargs.get('headless', False):
            self.sb.driver.maximize_window()
        
        vw, vh = self.getViewportSize()
        self._mouse_position = random.uniform(0, vw), random.uniform(0, vh)
        
    def stop(self):
        if hasattr(self, 'sb_context'):
            self.sb_context.__exit__(None, None, None)
        
    def wait(self, lower=0.5, upper=3.0) -> float:
        """Return a randomized delay sampled from a truncated normal distribution."""
        mu = (lower + upper) / 2
        sigma = (upper - lower) / 4
        d = random.gauss(mu, sigma)
        time.sleep(max(lower, min(d, upper)))
        return max(lower, min(d, upper))
    
    def isXpath(self, selector: str) -> bool:
        """Quick heuristic to detect XPath-like selectors."""
        s = selector.strip()
        return s.startswith(("/", "//", ".//", "("))
    def getElement(self, selector: str):
        """Find element using XPath if selector looks like XPath, otherwise CSS."""
        if self.isXpath(selector):
            return self.sb.driver.find_element(By.XPATH, selector)
        return self.sb.driver.find_element(By.CSS_SELECTOR, selector)
    
    def waitForPresence(self, selector: str, timeout: int = 10):
        """Wait for presence of element supporting CSS or XPath selectors."""
        return WebDriverWait(self.sb.driver, timeout).until(lambda d: self.getElement(selector))
    def waitForClickable(self, selector: str, timeout: int = 10):
        """Wait for an element to be clickable (best-effort)."""
        if self.isXpath(selector):
            return WebDriverWait(self.sb.driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
        return WebDriverWait(self.sb.driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
        )
    
    def _quadratic_bezier(self, p0, p1, p2, t):
        """Quadratic bezier point for t in [0,1]."""
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return x, y
    
    def _bezier_path(self, start, end, steps=12, wobble=0.2):
        """Return list of points between start and end (adds a random control point)."""
        cx = (start[0] + end[0]) / 2 + random.uniform(-wobble, wobble) * abs(end[0] - start[0])
        cy = (start[1] + end[1]) / 2 + random.uniform(-wobble, wobble) * abs(end[1] - start[1])
        control = (cx, cy)
        points = []
        for i in range(steps):
            t = i / float(steps - 1)
            points.append(self._quadratic_bezier(start, control, end, t))
        return points
    
    def viewportToScreen(self, x, y):
        # browser window position on the screen
        window_pos = self.sb.execute_script("return [window.screenX, window.screenY];")
        # border + title bar (approximate)
        outer_offsets = self.sb.execute_script(
            "return [window.outerWidth - window.innerWidth, window.outerHeight - window.innerHeight];"
        )
        screen_x = int(x + window_pos[0] + outer_offsets[0] / 2)
        screen_y = int(y + window_pos[1] + outer_offsets[1] - (outer_offsets[0]/2))  # approximate
        return screen_x, screen_y

    def _moveAlongPath(self, points, pause_per_step=(0.006, 0.02), rupture_point_range=(0.3, 0.7), spread_range=(0.15, 0.25)):
        """Move the real mouse cursor along the points list using ActionChains."""
        # We'll use the page <body> as a reference frame and move to offsets from it.
        body = self.sb.driver.find_element(By.TAG_NAME, "body")
        
        n = len(points)
        if n < 2:
            return
        # Gaussian timing profile
        mu = random.uniform(*rupture_point_range)  # rupture point somewhere between 30–70%
        sigma = random.uniform(*spread_range)  # spread controls how sharp/smooth the peak is
        t = np.linspace(0, 1, n)
        weights = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
        
        # Normalize to [min_pause, max_pause]
        w = weights / weights.max()  # scale to [0,1]
        min_pause, max_pause = pause_per_step
        pauses = min_pause + (1 - w) * (max_pause - min_pause)
        
        vw, vh = self.getViewportSize()
        
        for (pt, pause) in zip(points, pauses):
            x, y = int(round(pt[0])), int(round(pt[1]))
            x = max(1, min(vw - 2, x))
            y = max(1, min(vh - 2, y))
            try:
                sx, sy = self.viewportToScreen(x, y)
                pyautogui.moveTo(sx, sy)
                time.sleep(pause)
                self._mouse_position = (x, y)
            except WebDriverException:
                # best-effort move; ignore intermittent failures
                pass
    
    def move(self, selector: str, wait=(0.05, 0.25), pause_per_step=(0.006, 0.02), rupture_point_range=(0.3, 0.7), spread_range=(0.15, 0.25)):
        el = self.waitForPresence(selector)
        
        left, top, width, height = self.getElementClientRect(el)

        # choose a random target inside the element (viewport coords)
        target_x = left + random.uniform(0.15, 0.85) * width
        target_y = top + random.uniform(0.15, 0.85) * height
        
        start = self._mouse_position
        end = (target_x, target_y)
        path = self._bezier_path(start, end, steps=random.randint(8, 18), wobble=0.25)
        self._moveAlongPath(path, pause_per_step, rupture_point_range, spread_range)
        self.wait(*wait)
        
    def click(self, selector: str, wait=(0.05, 0.3), pause_per_step=(0.006, 0.02), rupture_point_range=(0.3, 0.7), spread_range=(0.15, 0.25)):
        self.scrollToElemen(selector)
        el = self.waitForPresence(selector)
        self.move(selector, wait=wait, pause_per_step=pause_per_step, rupture_point_range=rupture_point_range, spread_range=spread_range)

        try:
            # ActionChains(self.sb.driver).click().perform()
            pyautogui.click()
        except (ElementClickInterceptedException, WebDriverException):
            # fallback: JavaScript click as a last resort (still legitimate for testing)
            try:
                print('Click ERROR: try with JS')
                self.sb.execute_script("arguments[0].click();", el)
            except Exception as e:
                print('Click ERROR with JS')
                traceback.print_exc()
                pass
        self.wait(*wait)
    # def human_hover(self, selector: str, duration_range=(0.5, 2.0)):
    #     """Move to element and hold the cursor there for a human-like duration."""
    #     el = self.getElement(selector)
    #     loc = el.location
    #     size = el.size
    #     target = (int(loc['x'] + size['width'] * 0.5), int(loc['y'] + size['height'] * 0.5))
    #     # compute short path from center of viewport to target
    #     start = (self.sb.execute_script("return window.pageXOffset || 0") + self.sb.execute_script("return window.innerWidth") / 2,
    #              self.sb.execute_script("return window.pageYOffset || 0") + self.sb.execute_script("return window.innerHeight") / 2)
    #     path = self._bezier_path(start, target, steps=random.randint(6, 12), wobble=0.15)
    #     self._move_along_path(path, pause_per_step=random.uniform(0.008, 0.02))
    #     self.wait(duration_range)
    # def human_scroll(self, dy_range=(-400, 400)):
    #     """Perform a small, natural-feeling scroll."""
    #     dy = int(random.uniform(dy_range[0], dy_range[1]))
    #     # smooth scroll in small steps
    #     steps = max(3, int(abs(dy) / 50))
    #     for i in range(steps):
    #         part = int(dy / steps)
    #         self.sb.execute_script(f"window.scrollBy(0, {part});")
    #         time.sleep(random.uniform(0.02, 0.12))
    #     # final small pause
    #     self.wait((0.05, 0.3))
    
    def type(self, selector: str, text: str, clear: bool = True, wait=(0.08, 0.2), delay_range=(0.04, 0.22), mistake_chance=0.03, fix_delay=(0.08, 0.18)):
        """
        Click into field and type text char-by-char with randomized delays.
        Occasionally injects a small typo and corrects it to emulate human typing.
        """
        el = self.waitForPresence(selector)

        try:
            self.click(selector)
        except WebDriverException:
            try:
                el.click()
            except Exception:
                pass
        self.wait(*wait)
        
        if clear:
            current_val = el.get_attribute("value") or ""
            if current_val.strip():
                # simulate ctrl+a + backspace
                el.send_keys(Keys.CONTROL, "a")
                time.sleep(self.wait(0.04, 0.1))
                el.send_keys(Keys.BACKSPACE)
                time.sleep(self.wait(0.05, 0.15))
        for ch in text:
            # chance to make a typo
            if random.random() < mistake_chance and ch.isalnum():
                wrong = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                el.send_keys(wrong)
                time.sleep(random.uniform(*fix_delay))
                # backspace once to remove wrong char
                el.send_keys(Keys.BACKSPACE)
                time.sleep(random.uniform(*fix_delay))
                
            # send intended char
            if ch == '\n':
                el.send_keys(Keys.ENTER)
            else:
                el.send_keys(ch)
            time.sleep(random.uniform(delay_range[0], delay_range[1]) * (0.8 + random.random() * 0.6))
        # small pause after finishing typing
        self.wait(*wait)
        
    def paste(self, selector: str, text: str, clear: bool = True, wait=(0.08, 0.2)):
        """
        Click into field and paste text using clipboard (Ctrl+V) for faster input.
        Optionally clears existing content first.
        """
        el = self.waitForPresence(selector)

        try:
            self.click(selector)
        except WebDriverException:
            try:
                el.click()
            except Exception:
                pass

        self.wait(*wait)

        if clear:
            current_val = el.get_attribute("value") or ""
            if current_val.strip():
                el.send_keys(Keys.CONTROL, "a")
                self.wait(0.04, 0.1)
                el.send_keys(Keys.BACKSPACE)
                self.wait(0.05, 0.15)

        # copy text to clipboard and paste
        pyperclip.copy(text)
        el.send_keys(Keys.CONTROL, "v")

        self.wait(*wait)
        
    def scroll(self, direction=1, distance=(120, 240), wait=(0.05, 0.25)):
        dist = random.randint(*distance)
        
        if isinstance(distance[0], float):
            _, vh = self.getViewportSize()
            dist *= vh

        self.sb.execute_script(f"window.scrollBy(0, {dist * direction});")
        self.wait(*wait)
        
    def scrollToElemen(self, selector: str, distance=(120, 240), wait=(0.05, 0.25), max_scrolls=30):
        """
        Scrolls the page gradually and 'human-like' until the element is visible in viewport.
        """
        el = self.waitForPresence(selector)
        for i in range(max_scrolls):
            left, top, width, height = self.getElementClientRect(el)
            vw, vh = self.getViewportSize()

            # Check if fully in viewport
            if 0 <= top < vh - height and 0 <= left < vw - width:
                return True

            # scroll
            if top > vh:   # element is below viewport
                self.scroll(direction=1, distance=distance, wait=wait)
            elif top + height < 0:  # element is above viewport
                self.scroll(direction=-1, distance=distance, wait=wait)
            else:
                # partially visible, break
                break

        # final ensure it's visible (native scrollIntoView but smoother)
        try:
            self.sb.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", el)
        except Exception:
            pass
        return False
        
    def addCursor(self, color: str = "red", size: int = 8):
        """
        Inject a minimal dot that replaces the native cursor for debugging pointer paths.
        Use remove_cursor_dot(sb) to remove it.
        """
        js = f"""
            (function(){{
            if(document.getElementById('__cursor_dot__')) return;
            var dot = document.createElement('div');
            dot.id = '__cursor_dot__';
            dot.style.position = 'fixed';
            dot.style.width = '{size}px';
            dot.style.height = '{size}px';
            dot.style.background = '{color}';
            dot.style.borderRadius = '50%';
            dot.style.zIndex = '2147483647';
            dot.style.pointerEvents = 'none';
            dot.style.transform = 'translate(-50%, -50%)';
            document.body.appendChild(dot);
            document.body.style.cursor = 'none';
            document.addEventListener('mousemove', function(e){{
                dot.style.left = e.clientX + 'px';
                dot.style.top = e.clientY + 'px';
            }}, true);
            window.__remove_cursor_dot = function(){{ var d=document.getElementById('__cursor_dot__'); if(d) d.remove(); document.body.style.cursor = ''; }};
            }})();
        """
        self.sb.driver.execute_script(js)
    def removeCursor(self):
        js = "if(window.__remove_cursor_dot){ try{ window.__remove_cursor_dot(); delete window.__remove_cursor_dot; } catch(e){} }"
        self.sb.driver.execute_script(js)
        
    def saveCookies(self, path='./'):
        path = Path(path)
        cookies = self.sb.driver.get_cookies()
        if not path.is_file():
            path = path / "cookies.pkl"
        with open(path.as_posix(), "wb") as f:
            pickle.dump(cookies, f)
            
    def loadCookies(self, path='./', not_exist_ok=True):
        path = Path(path)
        if not path.is_file():
            path = path / "cookies.pkl"
        if not path.exists():
            if not_exist_ok is False:
                raise Exception(f'The file {path} do not exist.')
            else:
                return
        with open(path, "rb") as f:
            cookies = pickle.load(f)
        self.sb.driver.delete_all_cookies()
        for cookie in cookies:
            self.sb.driver.add_cookie(cookie)
        self.sb.driver.refresh() 
        
    def getViewportSize(self):
        """Return (width, height) of the viewport (innerWidth, innerHeight)."""
        w = self.sb.execute_script("return window.innerWidth")
        h = self.sb.execute_script("return window.innerHeight")
        return int(w), int(h)

    def getElementClientRect(self, el):
        """Return (left, top, width, height) of element relative to viewport."""
        rect = self.sb.execute_script(
            "var r = arguments[0].getBoundingClientRect();"
            "return {left: r.left, top: r.top, width: r.width, height: r.height};",
            el,
        )
        return float(rect["left"]), float(rect["top"]), float(rect["width"]), float(rect["height"])
    
    def closeGoogleTranslatePopup(self, wait=(1.0, 2.0)):
        """
        Try to close the Google Translate popup if present.
        """
        self.wait(*wait)
        pyautogui.press("esc")
        self.wait(*wait)

