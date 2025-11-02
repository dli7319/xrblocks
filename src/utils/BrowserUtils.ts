export function onDesktopUserAgent() {
  return !/Mobi|Android|iPhone/i.test(navigator.userAgent);
}
