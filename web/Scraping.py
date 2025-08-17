
import asyncio
import random
import math
import requests
from bs4 import BeautifulSoup
import string
from tzlocal import get_localzone
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import json
import platform
from typing import Literal
import locale


from py_libraries.web.UserAgentProvider import UserAgentProvider
from py_libraries.string import toCamelCase

class Scraping:
    """
    Tips:
        - Get the proxies from the same regions than your defined local time
    
    To verify manually: playwright codegen https://www.redbubble.com/auth/login
    """
    def __init__(self,
                 proxies: list[str] = None,
                 headless: bool = True,
                 max_pages: int = 3):
        """
        proxies: optional initial list of proxy URLs
        max_pages: max concurrent pages to keep open
        """
        self.proxies = proxies or []
        self.headless = headless
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.pages: list[Page] = []
        self.mouses_position = []
        self.max_pages = max_pages

        # pool to limit concurrent page actions if needed
        self._semaphore = asyncio.Semaphore(max_pages)

    async def start(self, 
                    proxy: str = None,
                    user_agent: str = None,
                    viewport: tuple = (1920, 1080),
                    locale_time: str = None,
                    timezone_id: str = None
                    ):
        """Initialize Playwright, launch browser (with optional proxy) and create a stealth context."""
        await self.stop()
        self.playwright = await async_playwright().start()
        self.proxy = proxy
        
        # Define the browser
        launch_args = {"headless": self.headless}
        if proxy:
            launch_args["proxy"] = {"server": proxy}
        self.browser = await self.playwright.chromium.launch(**launch_args)

       # If not proxy, use real infos to avoid suspision
        if self.headless is True:
            self.os=None
            self.locale_time= random.choice(Scraping.localeTime()) if locale_time is None else locale_time
            self.timezone_id=random.choice(Scraping.timeZones()) if timezone_id is None else timezone_id
            
            # set up the context
            user_agent_provider = UserAgentProvider()
            user_agent_provider.loadDefault()
            self.user_agent_list = user_agent_provider.get(browser='chrome', device='desktop', os=self.os)
            self.user_agent = user_agent if user_agent else random.choice(self.user_agent_list)
            self.width, self.height = viewport
            self.context = await self.browser.new_context(
                viewport={"width": self.width, "height": self.height},
                user_agent=self.user_agent,
                locale=self.locale_time,
                timezone_id=self.timezone_id,
                device_scale_factor=1.0,
                is_mobile=False,
                has_touch=False,
            )
            # stealth init scripts
            await self._apply_stealth_scripts(self.context)
        else:
            self.context = await self.browser.new_context()
        
        

        # open initial page
        page = await self.context.new_page()
        self.pages = [page]
        self.mouses_position = [self._newMousePosition(page)]
        return page
    
    def _newMousePosition(self, page: Page):
        return random.uniform(0, page.viewport_size["width"]), random.uniform(0, page.viewport_size["height"])
    
    def isStarted(self) -> bool:
        """Check whether the browser and context are initialized and running."""
        return self.playwright is not None and self.browser is not None and self.context is not None
    
    @staticmethod
    def timeZones():
        return [
                "America/New_York",
                "Europe/London",
                "Europe/Paris",
                "Asia/Tokyo"
            ]
        
    @staticmethod
    def localeTime():
        return [
                "en-US",
                "en-GB",
                "fr-FR",
                "de-DE"
            ]

    async def rotateProxy(self, test=True, verbose=0, **kwargs):
        """Shut down existing context/browser and restart using next proxy in list."""
        end = (' ' * 20 + '\r') if verbose == 1 else ''
        await self.stop()
        if not self.proxies:
            raise RuntimeError("No proxies available to rotate to.")
        # pop next proxy (round robin)
        proxy = self.proxies.pop(0)
        if test:
            i=1
            while not Scraping.testProxy(proxy):
                if verbose >= 1:
                    print(f'{proxy} failed the test. {i}', end=end)
                if len(self.proxies) == 0:
                    raise RuntimeError("No proxies passing the test.")
                i+=1
                proxy = self.proxies.pop(0)
        
        self.proxies.append(proxy)
        
        return await self.start(proxy=proxy, **kwargs)

    async def newPage(self) -> Page:
        """Open a new page in the same context, respecting max_pages."""
        async with self._semaphore:
            if not self.context:
                raise RuntimeError("Context not started; call .start() first.")
            page = await self.context.new_page()
            self.pages.append(page)
            self.mouses_position.append(self._newMousePosition(page))
            return page

    async def closePage(self, page: Page):
        """Close one of the managed pages."""
        await page.close()
        page_idx = self.pages.index(page)
        self.pages.remove(page)
        self.mouses_position.pop(page_idx)
        
        
    async def goTo(self, page: Page, url:str, wait_until:Literal[ "load", "domcontentloaded", "networkidle", "commit"]="networkidle",
                        selector=None, timeout=8000):
        await page.goto(url, wait_until=wait_until)
        if selector:
            await page.wait_for_selector(selector, timeout=timeout)

    async def stop(self):
        """Cleanly shut down all pages, context, browser, and playwright."""
        for p in list(self.pages):
            await p.close()
        self.pages.clear()
        self.mouses_position.clear()
        if self.context:
            await self.context.close()
            self.context = None
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

    async def wait(self, a: float = 0.5, b: float = 2.0):
        """Random delay between a and b seconds."""
        await asyncio.sleep(random.uniform(a, b))
        
    async def type(self, page: Page, selector: str, text: str, delay_range=(0.05, 0.2), error_chance=0.05, focus_first=True, clear_first=False):
        locator = page.locator(selector)
        
        if focus_first:
            await self.click(page, selector)

        if clear_first:
            existing = await locator.input_value()
            if existing.strip():
                await locator.click(click_count=3)
                await self.wait(0.05, 0.1)
                await page.keyboard.press("Backspace")
                await self.wait(0.05, 0.1)

        typed = ""
        for char in text:
            if random.random() < error_chance:
                # Type wrong character
                wrong_char = random.choice(string.ascii_letters)
                await page.keyboard.type(wrong_char)
                await self.wait(delay_range[0]*2, delay_range[1]*2)
                # Press backspace to correct
                await page.keyboard.press("Backspace")
                await self.wait(*delay_range)

            await page.keyboard.type(char)
            typed += char
            await self.wait(*delay_range)

        await self.wait()
        
    def getMousePosition(self, page):
        if self.mouses_position is None:
            raise RuntimeError("Context not started; call .start() first.") 
        
        if page in self.pages:
            idx = self.pages.index(page)
        else:
            if isinstance(page, Page):
                self.pages.append(page)
                self.mouses_position.append(self._newMousePosition(page))
                idx = self.pages.index(page)
            else:
                raise AttributeError(f'page must be of type Page but is {type(page)}')
        
        return self.mouses_position[idx]
    
    def setMousePosition(self, page, x, y):
        if self.mouses_position is None:
            raise RuntimeError("Context not started; call .start() first.") 
        
        if page in self.pages:
            idx = self.pages.index(page)
        else:
            if isinstance(page, Page):
                self.pages.append(page)
                self.mouses_position.append(self._newMousePosition(page))
                idx = self.pages.index(page)
            else:
                raise AttributeError(f'page must be of type Page but is {type(page)}')
            
        self.mouses_position[idx] = [x, y]
    
    def displayMouseMovment(self, page: Page, start, end, show=True, path=None, factor=0.1):
        from py_libraries.visualize import Visualize
        import numpy as np
        
        points = np.array([list(p) for p in Scraping._bezier_curve(start, end, steps=random.randint(20, 40))])
        x, y = points[:, 0], points[:, 1]
        Visualize.Plot.plot((x * factor, y * factor), show=show, path=path, figsize=(page.viewport_size["width"] * factor, page.viewport_size["height"] * factor))
        
    async def mouseToSelector(self, page: Page, selector: str):
        locator = page.locator(selector)

        await locator.scroll_into_view_if_needed()

        box = await locator.bounding_box()
        
        if not box:
            raise RuntimeError(f"{selector} not visible")

        start = self.getMousePosition(page)
        
        end = (random.uniform(box["x"], box["x"] + box["width"]), random.uniform(box["y"], box["y"] + box["height"]/2))
        
        path = list(Scraping._bezier_curve(start, end, steps=random.randint(20, 40)))
        await page.mouse.move(path[0][0], path[0][1])
        for x, y in path[1:]:
            await page.mouse.move(x, y, steps=1)
            self.setMousePosition(page, x, y)
        await self.wait(0.1, 0.3)
        
        return end

    async def click(self, page: Page, selector: str):
        """Move along a Bézier curve to element, then click."""
        end = await self.mouseToSelector(page, selector)
        await page.mouse.click(end[0], end[1])
        await self.wait()

    async def scroll(self, page: Page,
                     distance_range=(0.2, 1.0),
                     wait_range=(0.1, 1.0)):
        """Random scroll by wheel events."""
        height = await page.evaluate("() => window.innerHeight")
        dist = random.uniform(*distance_range) * height
        await page.mouse.wheel(0, dist)
        await self.wait(*wait_range)

    @staticmethod
    def _bezier_curve(start, end, steps=25):
        cx = (start[0]+end[0])/2 + random.uniform(-80, 80)
        cy = (start[1]+end[1])/2 + random.uniform(-80, 80)
        for t in [i/steps for i in range(steps+1)]:
            x = (1-t)**2*start[0] + 2*(1-t)*t*cx + t**2*end[0]
            y = (1-t)**2*start[1] + 2*(1-t)*t*cy + t**2*end[1]
            yield x , y
            
    async def isSelector(self, page, selector):
        count = await page.locator(selector).count()
        return count > 0

    # ─── PROXY MANAGEMENT ──────────────────────────────────────────────────────
    @staticmethod
    def testProxy(proxy: str, url="https://www.google.com", timeout=5):
        try:
            r = requests.get(url,
                             proxies={"http":proxy,"https":proxy},
                             timeout=timeout)
            return r.status_code == 200
        except:
            return False
        
    @staticmethod
    def getGeonodeInfos(limit:int=100, sort_by:str='lastChecked', sort_type:str='desc', 
                            anonymityLevel:str=None, country:str=None, filterPort:int=None, protocols:str=None, 
                            speed:int=None, filterUpTime:int=None, filterLastChecked:int=None, google:str=None) -> list[str]:
        """https://geonode.com/free-proxy-list"""
        args = [f"{k}={v}" for k, v in locals().items() if v is not None]
        args = '?' + '&'.join(args) if len(args) > 0 else ''
        url = f"https://proxylist.geonode.com/api/proxy-list{args}"
        return requests.get(url).json().get("data", [])

    @staticmethod
    def getGeonodeProxies(limit:int=100, sort_by:str='lastChecked', sort_type:str='desc', 
                            anonymityLevel:str=None, country:str=None, filterPort:int=None, protocols:str=None, 
                            speed:int=None, filterUpTime:int=None, filterLastChecked:int=None, google:str=None) -> list[str]:
        geonode_infos = Scraping.getGeonodeInfos(**locals())
        return [f"http://{d['ip']}:{d['port']}" for d in geonode_infos]
    
    @staticmethod
    def getFreeProxyInfos(limit:int=300, path_extension:str='', **kwargs) -> list[str]:
        """Scrape SSLProxies.org for fast list."""
        r = requests.get("https://free-proxy-list.net"+path_extension)
        soup = BeautifulSoup(r.text, "html.parser")
        header = [toCamelCase(th.text.strip()) for th in soup.select("table thead tr th")]
        
        rows = []
        for row in soup.select("table tbody tr"):
            row_data = {key: td.text.strip() for key, td in zip(header, row.find_all("td"))}
            # Apply filtering
            if all(row_data.get(k) == str(v) for k, v in kwargs.items()):
                rows.append(row_data)
                if len(rows) == limit:
                    break
                
        return rows

    @staticmethod
    def getFreeProxyProxies(limit:int=300, path_extension:str='', **kwargs) -> list[str]:
        """Scrape SSLProxies.org for fast list."""
        rows = Scraping.getFreeProxyInfos(limit=limit, path_extension=path_extension, **kwargs)
        
        return [f"http://{row['ipAddress']}:{row['port']}" for row in rows]
    
    @staticmethod
    def getProxyScrapeProxies(limit:int=100, timeout:int=10000, protocol:Literal['http', 'socks4', 'socks5', 'all']=None,
                        country:str=None, ssl:Literal['all','yes','no']=None, anonymity:Literal['elite','anonymous','transparent','all']=None,
                        skip:int=None):
        """https://docs.proxyscrape.com/"""
        args = [f"{k}={v}" for k, v in locals().items() if v is not None]
        args = '&' + '&'.join(args) if len(args) > 0 else ''
        api="https://api.proxyscrape.com/v4/free-proxy-list/get?request=displayproxies" + args
        r = requests.get(api)
        return [f"http://{line.strip()}" for line in r.text.splitlines() if line]
    

    def reloadProxyPool(self, pool_size=50, test_proxies=False, geonode=True, free_proxy=True, proxy_scraper=True, geonode_kwargs={}, free_proxy_kwargs={}, proxy_scraper_kwargs={}):
        """Fetch from all sources into self.proxies."""
        def updateLimit(dc):
            if 'limit' not in dc:
                dc['limit'] = pool_size
                
        def getProxyList(is_desire, function, kwargs):
            if is_desire is False:
                return []
            return function(**kwargs)
        
        pool = []
        
        # update limits
        updateLimit(geonode_kwargs)
        updateLimit(free_proxy_kwargs)
        updateLimit(proxy_scraper_kwargs)
        
        pool += getProxyList(geonode, Scraping.getGeonodeProxies, geonode_kwargs)
        pool += getProxyList(free_proxy, Scraping.getFreeProxyProxies, free_proxy_kwargs)
        pool += getProxyList(proxy_scraper, Scraping.getProxyScrapeProxies, proxy_scraper_kwargs)
        
        random.shuffle(pool)

        self.proxies = [p for p in pool if not test_proxies or Scraping.testProxy(p)][:pool_size]
        
    @staticmethod
    def proxyInfos(proxy):
        proxy = proxy.replace('http://','').split(':')[0]
        return requests.get(f"http://ip-api.com/json/{proxy}").json()

    async def _apply_stealth_scripts(self, ctx: BrowserContext):
        # override webdriver
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        # mock plugins
        await ctx.add_init_script(
            "window.navigator.plugins = [1,2,3,4,5];"
        )
        # mock languages
        await ctx.add_init_script(
            "Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});"
        )


