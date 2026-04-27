export function isIndianMarketOpen(): boolean {
  const now = new Date();
  
  // Convert to IST (UTC + 5:30)
  const istOffset = 5.5 * 60 * 60 * 1000;
  const istTime = new Date(now.getTime() + now.getTimezoneOffset() * 60000 + istOffset);
  
  const day = istTime.getDay(); // 0 = Sunday, 1 = Monday, etc.
  const hours = istTime.getHours();
  const minutes = istTime.getMinutes();
  
  // Weekend check
  if (day === 0 || day === 6) return false;
  
  // 9:15 AM to 3:30 PM (15:30)
  const timeInMinutes = hours * 60 + minutes;
  const marketOpen = 9 * 60 + 15;  // 555
  const marketClose = 15 * 60 + 30; // 930
  
  return timeInMinutes >= marketOpen && timeInMinutes <= marketClose;
}