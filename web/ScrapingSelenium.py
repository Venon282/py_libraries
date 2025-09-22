
import random
import time
from typing import Optional
import numpy as np
import scipy.interpolate as si
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
from selenium_stealth import stealth
from seleniumbase import SB

class ScrapingSelenium:
    """
    Lightweight Selenium wrapper using undetected-chromedriver + selenium-stealth.
    Includes random pauses & simple human-like mouse moves.
    """
    def __init__(self,
                 headless: bool = True
                 ):
        self.headless = headless
        self.driver = None
        self.mouse_pos = None

    def start(self):
        self.options = webdriver.ChromeOptions() 
        self.options.headless = self.headless
        self.options.add_argument("start-maximized")

        self.driver = uc.Chrome(options=self.options)

        self.mouse_pos = (         
            random.uniform(0, self.viewportWidth()), 
            random.uniform(0, self.viewportHeight())
            )
        
        self.wait(2, 3)

    def verifyStart(self):
        if self.driver is None:
            raise RuntimeError("Call the start() method before.")

    def stop(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            finally:
                self.driver = None

    def wait(self, a=0.5, b=2.0):
        time.sleep(random.uniform(a, b))

    def move(self, start, end, wait_range=(0.01, 0.035), steps=(60,120)):
        # build control points between start & end
        ctrl = [start]
        for i in range(1,4):
            t = i/4
            jitter =  random.uniform(-50,50)
            ctrl.append((
                start[0] + (end[0]-start[0])*t + jitter,
                start[1] + (end[1]-start[1])*t + jitter
            ))
        ctrl.append(end)

        # generate spline path & move
        path = self._generate_spline(ctrl, steps=random.randint(*steps))
        ac = ActionChains(self.driver)

        # first jump to path[0]
        dx, dy = path[0]
        ac.move_by_offset(dx, dy).perform()
        px, py = dx, dy
        self.mouse_pos = px, py

        for x,y in path[1:]:
            self.mouse_pos = x, y
            ac.move_by_offset(x-px, y-py).perform()
            px, py = x, y
        self.wait(*wait_range)


    def click(self, element, distance_scroll_range=(0.2, 1.0), wait_scroll_range=(0.1, 1.0), max_scroll_attempt=50):
        if self.scrollToElement(element, distance_range=distance_scroll_range, wait_range=wait_scroll_range, max_attempt=max_scroll_attempt) is False:
            raise RuntimeError('Impossible to scroll until the element.')

        # get current and futur mouse points
        start = self.mouse_pos
        end = self.elementRandomPoint(element)

        self.move(start, end, wait_range=(0.01, 0.035))

        ActionChains(self.driver).click().perform()

    
    def elementRandomPoint(self, element, rate=0.9):
        rect = element.rect
        width = rect['width']
        height = rect['height']
        x = rect['x']
        y = rect['y']

        # Define margin to avoid the outer 5% on each side (for rate central area)
        margin_rate = (1 - rate) / 2
        x_margin = width * margin_rate
        y_margin = height * margin_rate

        # Generate random point inside the rate central area
        rand_x = random.uniform(x + x_margin, x + width - x_margin)
        rand_y = random.uniform(y + y_margin, y + height - y_margin)

        return rand_x, rand_y
    
    def scroll(self, direction=1, distance_range=(0.2, 1.0), wait_range=(0.1, 1.0)):
        """Random scroll by wheel events."""
        height = self.viewportHeight()
        dist = random.uniform(*distance_range) * height * direction
        ActionChains(self.driver).scroll_by_amount(0, dist).perform()
        self.wait(*wait_range)

    def scrollToElement(self, element, distance_range=(0.2, 1.0), wait_range=(0.6, 2.0), max_attempt=50):
        while self.isElementVisibleInViewport(element) is False and max_attempt > 0:
            if self.isElementAboveViewport(element):
                direction = -1
            elif self.isElementBelowViewport(element):
                direction = 1
            else: 
                return True

            self.scroll(direction=direction)
            max_attempt -= 1

        self.wait(*wait_range)
        return max_attempt > 0

    def viewportHeight(self):
        return self.driver.execute_script("""
            return window.innerHeight || document.documentElement.clientHeight;
        """)

    def viewportWidth(self):
        return self.driver.execute_script("""
            return window.innerWidth || document.documentElement.clientWidth;
        """)   

    def isElementAboveViewport(self, element):
        return self.driver.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            return rect.bottom < 0;
        """, element)

    def isElementBelowViewport(self, element):
        return self.driver.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            return rect.top > (window.innerHeight || document.documentElement.clientHeight);
        """, element)

    def isElementAtLeastHalfAboveViewport(self, element):
        return driver.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            var elementHeight = rect.height;
            var visibleHeight = Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0);
            return visibleHeight < (elementHeight / 2);
        """, element)

    def isElementAtLeastHalfBelowViewport(self, element):
        return driver.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            var elementHeight = rect.height;
            var visibleHeight = Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0);
            return visibleHeight < (elementHeight / 2);
        """, element)

    def isElementVisibleInViewport(self, element: WebElement):
        return self.driver.execute_script("""
            var elem = arguments[0],
                box = elem.getBoundingClientRect(),
                cx = box.left + box.width / 2,
                cy = box.top + box.height / 2,
                e = document.elementFromPoint(cx, cy);
            for (; e; e = e.parentElement) {
                if (e === elem)
                    return true;
            }
            return false;
        """, element)

    @staticmethod
    def _bezier_curve(start, end, steps=25):
        cx = (start[0]+end[0])/2 + random.uniform(-80, 80)
        cy = (start[1]+end[1])/2 + random.uniform(-80, 80)
        for t in [i/steps for i in range(steps+1)]:
            x = (1-t)**2*start[0] + 2*(1-t)*t*cx + t**2*end[0]
            y = (1-t)**2*start[1] + 2*(1-t)*t*cy + t**2*end[1]
            yield x, y

    @staticmethod
    def _generate_spline(points, steps=50):
        pts = np.array(points)
        x, y = pts[:,0], pts[:,1]
        t = range(len(pts))
        ipl = np.linspace(0, len(pts)-1, steps)
        xt = si.splrep(t, x, k=3)
        yt = si.splrep(t, y, k=3)
        # pad the coef array
        xt = list(xt)
        xt[1] = x.tolist() + [0,0,0,0]

        yt = list(yt)
        yt[1] = y.tolist() + [0,0,0,0]

        xi = si.splev(ipl, xt)
        yi = si.splev(ipl, yt)

        return list(zip(xi, yi))

    def testCloudflareTurnstile(self, url='https://seleniumbase.io/apps/turnstile'):
        # https://2captcha.com/demo/cloudflare-turnstile
        self.verifyStart()
        # self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': """
        #     Element.prototype._attachShadow = Element.prototype.attachShadow;
        #     Element.prototype.attachShadow = function () {
        #         return this._attachShadow( { mode: "open" } );
        #     };
        #     """})
        self.driver.get(url)


        session = self.driver.execute_cdp_cmd
        # Get root document
        document = session("DOM.getDocument", {})

        # Find your closed shadow host nodeId via DOM.querySelector
        node = session("DOM.querySelector", {
            "nodeId": document["root"]["nodeId"],
            "selector": "div.cf-turnstile"
        })

        # Try to get the shadow root with DOM.describeNode
        shadow_info = session("DOM.describeNode", {"nodeId": node["nodeId"]})
        self.driver.execute_cdp_cmd("DOM.enable", {})
        flattened = self.driver.execute_cdp_cmd("DOM.getFlattenedDocument", {
            "depth": -1,  # unlimited
            "pierce": True  # pierce shadow DOM
        })
        print(flattened)
        # action = ActionChains(self.driver)
        # closed_shadow_host = self.driver.find_element(By.CSS_SELECTOR, "div.cf-turnstile")
        # shadow_root = self.driver.execute_script('return arguments[0].root.querySelector("#cf-chl-widget-pp0hq")', closed_shadow_host)

        # for f in shadow_root.find_elements(By.TAG_NAME, "iframe"):
        #     print("→ iframe:", f.get_attribute("id"), f.get_attribute("src"))
        # self.wait(3, 6)
        # self.displayPointer()

        # host = WebDriverWait(self.driver, 15).until(
        #     EC.visibility_of_element_located((By.CSS_SELECTOR, "div.cf-turnstile, .turnstile"))
        # )

        # self.driver.execute_script("arguments[0].style.border='3px solid red'", host)

        # # 2) Scroll dans la vue (facultatif mais recommandé)
        # self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", host)

        # # 3) Clic JavaScript direct sur le host
        # #    (Selenium envoie l’événement au bon endroit, shadow DOM fermé ou pas)
        # self.driver.execute_script("arguments[0].click();", host)

        # # 4) Optionnel : attendre que le token soit renseigné
        # WebDriverWait(self.driver, 10).until(
        #     lambda d: d.find_element(By.CSS_SELECTOR, "input[name='cf-turnstile-response']").get_attribute("value") != ""
        # )

    def displayPointer(self):
        cursor_js = """
        (function(){
          const d = document.createElement('div');
          d.id = 'mouse-tracker';
          Object.assign(d.style, {
            position: 'absolute',
            width: '10px',
            height: '10px',
            background: 'red',
            borderRadius: '50%',
            pointerEvents: 'none',
            zIndex: 2147483647
          });
          document.body.appendChild(d);
          document.addEventListener('mousemove', e => {
            d.style.left = e.pageX + 'px';
            d.style.top  = e.pageY + 'px';
          });
        })();
        """
        # Note: run this _after_ driver.get() if you want to see it on the landing page.
        self.driver.execute_script(cursor_js)

    def __del__(self):
        self.stop()


