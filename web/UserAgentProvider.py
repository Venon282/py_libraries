from typing import List, Dict, Union

class UserAgentProvider:
    def __init__(self, ua_list: List[Dict] = None):
        """
        ua_list : liste de dicts, chaque dict contient :
          - 'ua'       : chaîne user agent
          - 'browser'  : ex. "Chrome", "Firefox", "Safari"
          - 'device'   : ex. "desktop", "mobile", "tablet"
          - 'os'       : ex. "Windows", "macOS", "Linux", "Android", "iOS"
        """
        self.ua_list = ua_list or []

    def loadDefault(self):
        """Charge un petit jeu d'exemples (à étendre selon vos besoins)."""
        self.ua_list = [
            # === Desktop Chrome ===
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36",
                "browser": "Chrome", "device": "desktop", "os": "Windows"
            },
            {
                "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36",
                "browser": "Chrome", "device": "desktop", "os": "macOS"
            },
            {
                "ua": "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36",
                "browser": "Chrome", "device": "desktop", "os": "Linux"
            },

            # === Desktop Firefox ===
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) "
                    "Gecko/20100101 Firefox/116.0",
                "browser": "Firefox", "device": "desktop", "os": "Windows"
            },
            {
                "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:116.0) "
                    "Gecko/20100101 Firefox/116.0",
                "browser": "Firefox", "device": "desktop", "os": "macOS"
            },
            {
                "ua": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:116.0) "
                    "Gecko/20100101 Firefox/116.0",
                "browser": "Firefox", "device": "desktop", "os": "Linux"
            },

            # === Desktop Safari (WebKit) ===
            {
                "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/16.1 Safari/605.1.15",
                "browser": "Safari", "device": "desktop", "os": "macOS"
            },

            # === Desktop Edge ===
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36 Edg/116.0.1938.76",
                "browser": "Edge", "device": "desktop", "os": "Windows"
            },
            {
                "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36 Edg/116.0.1938.76",
                "browser": "Edge", "device": "desktop", "os": "macOS"
            },

            # === Desktop Opera ===
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36 OPR/101.0.4843.35",
                "browser": "Opera", "device": "desktop", "os": "Windows"
            },
            {
                "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36 OPR/101.0.4843.35",
                "browser": "Opera", "device": "desktop", "os": "macOS"
            },

            # === Desktop Vivaldi / Brave / Chromium ===
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Safari/537.36 Vivaldi/6.1",
                "browser": "Vivaldi", "device": "desktop", "os": "Windows"
            },
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Brave/1.69.132 Chrome/116.0.5845.96 Safari/537.36",
                "browser": "Brave", "device": "desktop", "os": "Windows"
            },
            {
                "ua": "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chromium/116.0.5845.96 Safari/537.36",
                "browser": "Chromium", "device": "desktop", "os": "Linux"
            },

            # === Legacy IE11 ===
            {
                "ua": "Mozilla/5.0 (Windows NT 10.0; Trident/7.0; rv:11.0) like Gecko",
                "browser": "IE", "device": "desktop", "os": "Windows"
            },

            # === Mobile Chrome (Android) ===
            {
                "ua": "Mozilla/5.0 (Linux; Android 13; SM-G991B) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Mobile Safari/537.36",
                "browser": "Chrome", "device": "mobile", "os": "Android"
            },

            # === Mobile Safari (iOS) ===
            {
                "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/16.5 Mobile/15E148 Safari/604.1",
                "browser": "Safari", "device": "mobile", "os": "iOS"
            },
            {
                "ua": "Mozilla/5.0 (iPad; CPU OS 16_5 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/16.5 Mobile/15E148 Safari/604.1",
                "browser": "Safari", "device": "tablet", "os": "iOS"
            },

            # === Mobile Edge / Opera Mini / Samsung Internet / UC Browser ===
            {
                "ua": "Mozilla/5.0 (Linux; Android 13; SM-G991B) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/116.0.5845.96 Mobile Safari/537.36 EdgA/116.0.1938.76",
                "browser": "Edge", "device": "mobile", "os": "Android"
            },
            {
                "ua": "Opera/9.80 (Android; Opera Mini/58.2.2254/132.22; U; en) "
                    "Presto/2.12.423 Version/12.16",
                "browser": "Opera Mini", "device": "mobile", "os": "Android"
            },
            {
                "ua": "Mozilla/5.0 (Linux; Android 13; SM-G991B) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "SamsungBrowser/23.0 Chrome/116.0.5845.96 Mobile Safari/537.36",
                "browser": "Samsung Internet", "device": "mobile", "os": "Android"
            },
            {
                "ua": "Mozilla/5.0 (Linux; U; Android 13; en-US; SM-G991B) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Version/4.0 UCBrowser/13.4.0.1305 Mobile Safari/537.36",
                "browser": "UC Browser", "device": "mobile", "os": "Android"
            },

            # === Bots / Crawlers ===
            {
                "ua": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                "browser": "Googlebot", "device": "bot", "os": "bot"
            },
            {
                "ua": "Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)",
                "browser": "Bingbot", "device": "bot", "os": "bot"
            },
            {
                "ua": "DuckDuckBot/1.0; (+http://duckduckgo.com/duckduckbot.html)",
                "browser": "DuckDuckBot", "device": "bot", "os": "bot"
            }
        ]

    def get(
        self,
        browser: Union[str, List[str]] = None,
        device: Union[str, List[str]] = None,
        os: Union[str, List[str]] = None
    ) -> List[str]:
        """
        Returns the list of UAs corresponding to the specified filters.
        The browser, device and os arguments can be either :
          - a character string (exact match, case insensitive)
          - a list of strings (match on one of the values)
          - None (no filtering on this field)
        """
        def norm_input(element):
            if element is None:
                return None
            if isinstance(element, list):
                return [e.lower() for e in element]
            return element.lower()

        b_filter = norm_input(browser)
        d_filter = norm_input(device)
        o_filter = norm_input(os)

        result = []
        for entry in self.ua_list:
            val_b = entry["browser"].lower()
            val_d = entry["device"].lower()
            val_o = entry["os"].lower()

            # Filtre browser
            if b_filter:
                if isinstance(b_filter, list):
                    if val_b not in b_filter: continue
                else:
                    if val_b != b_filter: continue
            # Filtre device
            if d_filter:
                if isinstance(d_filter, list):
                    if val_d not in d_filter: continue
                else:
                    if val_d != d_filter: continue
            # Filtre os
            if o_filter:
                if isinstance(o_filter, list):
                    if val_o not in o_filter: continue
                else:
                    if val_o != o_filter: continue

            result.append(entry["ua"])
        return result