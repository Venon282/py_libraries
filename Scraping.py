import asyncio
import random
import math
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import json

class Scraping:
    def __init__(self, proxy: str = None, headless: bool = True):
        self.proxy = proxy
        self.headless = headless
        self.playwright = None
        self.browser: Browser = None
        self.context: BrowserContext = None
        self.page: Page = None

    async def start(self):
        """Start Playwright, launch the browser (with optional proxy), and create a hardened context."""
        self.playwright = await async_playwright().start()
        launch_args = {"headless": self.headless}
        if self.proxy:
            launch_args["proxy"] = {"server": self.proxy}

        self.browser = await self.playwright.chromium.launch(**launch_args)
        self.user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=self.user_agent,
            locale="en-US",
            timezone_id="America/New_York",
            device_scale_factor=1.0,
            is_mobile=False,
            has_touch=False,
        )

        # Override webdriver flag
        await self.context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        # await self.context.add_init_script("""
        #     // Disable WebRTC IP leak
        #     Object.defineProperty(navigator, 'mediaDevices', { get: () => undefined });
        # """)

        self.page = await self.context.new_page()

    async def stop(self):
        """Cleanly close everything."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def wait(self, a: float = 0.5, b: float = 2.0):
        """Wait between a and b seconds."""
        await asyncio.sleep(random.uniform(a, b))

    async def human_type(self, selector: str, text: str, delay_range: tuple = (0.05, 0.2)):
        """Type text like a human, with per‑character random delay."""
        for char in text:
            await self.page.type(selector, char, delay=random.uniform(*delay_range))
        await self.random_wait(0.2, 0.5)

    def _bezier_curve(self, start, end, steps=25):
        """Generate a simple quadratic Bézier curve from start to end."""
        # control point randomly offset
        cx = (start[0] + end[0]) / 2 + random.uniform(-100, 100)
        cy = (start[1] + end[1]) / 2 + random.uniform(-100, 100)
        for t in [i/steps for i in range(steps+1)]:
            x = (1-t)**2 * start[0] + 2*(1-t)*t * cx + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * cy + t**2 * end[1]
            yield {"x": x, "y": y}

    async def human_mouse_move(self, selector: str):
        """Move mouse along a curve to a target element, then click."""
        box = await self.page.locator(selector).bounding_box()
        if not box:
            raise RuntimeError(f"Element {selector} not visible")
        start = (random.uniform(0, 1920), random.uniform(0, 1080))
        end = (box["x"] + box["width"]/2, box["y"] + box["height"]/2)
        path = list(self._bezier_curve(start, end, steps=random.randint(20, 40)))
        await self.page.mouse.move(path[0]["x"], path[0]["y"])
        for point in path[1:]:
            await self.page.mouse.move(point["x"], point["y"], steps=1)
        await self.random_wait(0.1, 0.3)
        await self.page.mouse.click(end[0], end[1])

    async def randomScroll(self, times: int = 3):
        """Perform small random scrolls up/down."""
        for _ in range(times):
            distance = random.uniform(-300, 300)
            await self.page.mouse.wheel(0, distance)
            await self.random_wait(0.3, 1.0)
            
    async def scroll(self, heigh_range=(0.1, 0.8), wait_range=(0.1, 1.0)):
        height = await self.page.evaluate("() => window.innerHeight")
        scroll_distance = random.uniform(*heigh_range) * height
        await self.page.evaluate(f"window.scrollBy(0, {scroll_distance})")
        await self.wait(*wait_range)
        
    @staticmethod
    def testProxy(proxy, url="https://www.google.com"):
        try:
            r = requests.get(url, proxies={"http": proxy, "https": proxy}, timeout=5)
            return r.status_code == 200
        except:
            return False
        
    @staticmethod
    def getGeonodeProxyList(args='google=true&limit=500&page=1&sort_by=lastChecked&sort_type=desc'):
        """https://geonode.com/free-proxy-list"""
        
        url = 'https://proxylist.geonode.com/api/proxy-list?' + args
        r = requests.get(url)
        datas = json.loads(r.text)
        return [f"http://{data['ip']}:{data['port']}"for data in datas['data']]
    
  
    @staticmethod
    def getGeonodeProxy(args='google=true&limit=500&page=1&sort_by=lastChecked&sort_type=desc'):
        proxies = Scraping.getGeonodeProxyList(args=args)
        while len(proxies) > 0:
            proxy = proxies.pop(random.randint(0, len(proxies)-1))
            if Scraping().testProxy(proxy):
                return proxy
        
    
    @staticmethod
    def getFreeProxyList(code=None, anonymity=None, google=None, https=None, infos=False, args=''):
        def verif(cols, pos, desire):
            if desire is None:
                return True
            
            return desire.strip().lower() == cols[pos]
        
        url = "https://free-proxy-list.net" + args
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        proxies = []

        for row in soup.select("table tbody tr"):
            cols = [td.text.strip().lower() for td in row.find_all("td")]
            if len(cols) < 7:
                continue
            
            if (verif(cols, 2, code)        and
                verif(cols, 4, anonymity)   and
                verif(cols, 5, google)      and
                verif(cols, 6, https)           ):
                
                ip = cols[0]
                port = cols[1]
                proxy = f"http://{ip}:{port}"
                proxies.append([proxy, cols])
            
        return proxies
    
    @staticmethod
    def getFreeProxy(code=None, anonymity=None, google=None, https=None, infos=False, args=''):
        proxies = Scraping.getFreeProxyList(code=code, anonymity=anonymity, google=google, https=https, infos=infos, args=args)
        while len(proxies) > 0:
            proxy = proxies.pop(random.randint(0, len(proxies)-1))
            if Scraping().testProxy(proxy if infos is False else proxy[0]):
                return proxy
    

# ——— Usage ———
async def main():
    scraper = Scraping(proxy="http://my.proxy.server:3128", headless=False)
    await scraper.start()
    try:
        html = await scraper.scrape("https://example.com")
        print("Scraped length:", len(html))
    finally:
        await scraper.stop()

if __name__ == "__main__":
    asyncio.run(main())
