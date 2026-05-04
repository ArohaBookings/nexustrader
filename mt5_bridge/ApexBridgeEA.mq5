#property strict

#include <Trade/Trade.mqh>

input string serverUrl = "http://127.0.0.1:8000";
input string apiKey = "";
input int pollSeconds = 2;
input int maxPositions = 50;
input int maxPerSymbol = 10;
input int spreadMaxPoints = 60;
input double killSwitchDailyLoss = 3.0;
input int slippage = 50;
input long magicNumber = 20260304;
input bool csvLogEnabled = true;
input int duplicateWindowSeconds = 600;

struct DecisionData
{
   string signal_id;
   string action;
   string target_ticket;
   string symbol;
   string side;
   double lot;
   double sl;
   double tp;
   int max_slippage_points;
   string reason;
   bool trailing_enabled;
   int trailing_points;
   int trailing_step_points;
   bool breakeven_enabled;
   int breakeven_trigger_points;
   double partial_rr;
   double partial_percent;
   string ai_summary;
};

struct PositionRule
{
   ulong ticket;
   string signal_id;
   bool trailing_enabled;
   int trailing_points;
   int trailing_step_points;
   bool breakeven_enabled;
   int breakeven_trigger_points;
   bool breakeven_done;
   bool partial_done;
   double partial_rr;
   double partial_percent;
   double initial_sl;
};

struct PendingHttpReport
{
   string key;
   string endpoint;
   string payload;
   datetime next_attempt;
   int attempts;
};

CTrade trade;
datetime g_lastPoll = 0;
datetime g_lastHeartbeat = 0;
bool g_backendReachable = true;
string g_lastErrorReason = "";
datetime g_lastDecisionAt = 0;
string g_lastDecisionId = "";
double g_dayStartEquity = 0.0;
int g_dayKey = 0;
string g_lastExecutionSummary = "none";
string g_executedSignals[];
string g_reportedDealIds[];
PositionRule g_rules[];
PendingHttpReport g_pendingReports[];

string TrimStr(string value)
{
   StringTrimLeft(value);
   StringTrimRight(value);
   return value;
}

string EscapeJson(string value)
{
   string output = value;
   StringReplace(output, "\\", "\\\\");
   StringReplace(output, "\"", "\\\"");
   StringReplace(output, "\r", " ");
   StringReplace(output, "\n", " ");
   return output;
}

void LogLine(string level, string message)
{
   string line = StringFormat("[ApexBridgeEA][%s][%s] %s", level, _Symbol, message);
   Print(line);
   if(!csvLogEnabled)
      return;
   int handle = FileOpen("ApexBridgeEA_log.csv", FILE_COMMON | FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   FileSeek(handle, 0, SEEK_END);
   FileWrite(handle, TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS), level, _Symbol, message);
   FileClose(handle);
}

string ExecutedFileName()
{
   return StringFormat("ApexBridgeEA_executed_%I64d_%s.csv", AccountInfoInteger(ACCOUNT_LOGIN), _Symbol);
}

string PendingReportsFileName()
{
   return StringFormat("ApexBridgeEA_pending_%I64d_%s.csv", AccountInfoInteger(ACCOUNT_LOGIN), _Symbol);
}

bool InStringArray(string &arr[], string value)
{
   int total = ArraySize(arr);
   for(int i = 0; i < total; i++)
   {
      if(arr[i] == value)
         return true;
   }
   return false;
}

void PushUnique(string &arr[], string value)
{
   if(value == "" || InStringArray(arr, value))
      return;
   int next = ArraySize(arr);
   ArrayResize(arr, next + 1);
   arr[next] = value;
}

void LoadExecutedSignals()
{
   ArrayResize(g_executedSignals, 0);
   int handle = FileOpen(ExecutedFileName(), FILE_COMMON | FILE_READ | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   while(!FileIsEnding(handle))
   {
      string value = TrimStr(FileReadString(handle));
      if(value != "")
         PushUnique(g_executedSignals, value);
   }
   FileClose(handle);
}

void RememberExecutedSignal(string signal_id)
{
   if(signal_id == "" || InStringArray(g_executedSignals, signal_id))
      return;
   PushUnique(g_executedSignals, signal_id);
   int handle = FileOpen(ExecutedFileName(), FILE_COMMON | FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   FileSeek(handle, 0, SEEK_END);
   FileWrite(handle, signal_id);
   FileClose(handle);
}

bool IsExecutedSignal(string signal_id)
{
   return InStringArray(g_executedSignals, signal_id);
}

int FindPendingReportIndex(string key)
{
   int total = ArraySize(g_pendingReports);
   for(int i = 0; i < total; i++)
   {
      if(g_pendingReports[i].key == key)
         return i;
   }
   return -1;
}

void SavePendingReports()
{
   int handle = FileOpen(PendingReportsFileName(), FILE_COMMON | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   int total = ArraySize(g_pendingReports);
   for(int i = 0; i < total; i++)
   {
      FileWrite(
         handle,
         g_pendingReports[i].key,
         g_pendingReports[i].endpoint,
         g_pendingReports[i].payload,
         g_pendingReports[i].attempts,
         (long)g_pendingReports[i].next_attempt
      );
   }
   FileClose(handle);
}

void LoadPendingReports()
{
   ArrayResize(g_pendingReports, 0);
   int handle = FileOpen(PendingReportsFileName(), FILE_COMMON | FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
      return;
   while(!FileIsEnding(handle))
   {
      string key = TrimStr(FileReadString(handle));
      string endpoint = TrimStr(FileReadString(handle));
      string payload = FileReadString(handle);
      int attempts = (int)FileReadNumber(handle);
      datetime nextAttempt = (datetime)((long)FileReadNumber(handle));
      if(key == "" || endpoint == "" || payload == "")
         continue;
      int next = ArraySize(g_pendingReports);
      ArrayResize(g_pendingReports, next + 1);
      g_pendingReports[next].key = key;
      g_pendingReports[next].endpoint = endpoint;
      g_pendingReports[next].payload = payload;
      g_pendingReports[next].attempts = attempts;
      g_pendingReports[next].next_attempt = nextAttempt;
   }
   FileClose(handle);
}

void RemovePendingReportAt(int index)
{
   int total = ArraySize(g_pendingReports);
   if(index < 0 || index >= total)
      return;
   for(int i = index; i < total - 1; i++)
      g_pendingReports[i] = g_pendingReports[i + 1];
   ArrayResize(g_pendingReports, total - 1);
   SavePendingReports();
}

void QueuePendingReport(string key, string endpoint, string payload, int attempts)
{
   int index = FindPendingReportIndex(key);
   int nextAttempts = MathMax(1, attempts);
   int delaySeconds = MathMin(60, MathMax(2, nextAttempts * 5));
   if(index < 0)
   {
      index = ArraySize(g_pendingReports);
      ArrayResize(g_pendingReports, index + 1);
   }
   g_pendingReports[index].key = key;
   g_pendingReports[index].endpoint = endpoint;
   g_pendingReports[index].payload = payload;
   g_pendingReports[index].attempts = nextAttempts;
   g_pendingReports[index].next_attempt = TimeCurrent() + delaySeconds;
   SavePendingReports();
}

string NormalizeBridgeSymbol(string value)
{
   string upper = StringToUpper(TrimStr(value));
   string normalized = "";
   int length = StringLen(upper);
   for(int i = 0; i < length; i++)
   {
      ushort c = StringGetCharacter(upper, i);
      bool isAlpha = (c >= 'A' && c <= 'Z');
      bool isDigit = (c >= '0' && c <= '9');
      if(isAlpha || isDigit)
         normalized += StringSubstr(upper, i, 1);
   }
   if(StringFind(normalized, "XAUUSD") == 0 || StringFind(normalized, "GOLD") == 0)
      return "XAUUSD";
   if(StringFind(normalized, "XAGUSD") == 0 || StringFind(normalized, "SILVER") == 0)
      return "XAGUSD";
   if(StringFind(normalized, "BTCUSD") == 0 || StringFind(normalized, "BTCUSDT") == 0 || StringFind(normalized, "XBTUSD") == 0)
      return "BTCUSD";
   if(StringFind(normalized, "DOGUSD") == 0 || StringFind(normalized, "DOGEUSD") == 0 || StringFind(normalized, "DOGE") == 0)
      return "DOGUSD";
   if(StringFind(normalized, "TRUMPUSD") == 0 || StringFind(normalized, "TRUMPUSDT") == 0 || StringFind(normalized, "TRUMP") == 0)
      return "TRUMPUSD";
   if(StringFind(normalized, "NAS100") == 0 || StringFind(normalized, "US100") == 0 || StringFind(normalized, "NASDAQ") == 0 || StringFind(normalized, "USTEC") == 0 || normalized == "NAS" || normalized == "NQ")
      return "NAS100";
   if(StringFind(normalized, "USOIL") == 0 || StringFind(normalized, "XTIUSD") == 0 || StringFind(normalized, "OILUSD") == 0 || normalized == "WTI" || normalized == "CL" || normalized == "OIL" || normalized == "USO")
      return "USOIL";
   if(StringFind(normalized, "EURUSD") == 0)
      return "EURUSD";
   if(StringFind(normalized, "GBPUSD") == 0)
      return "GBPUSD";
   if(StringFind(normalized, "USDJPY") == 0)
      return "USDJPY";
   if(StringFind(normalized, "AUDJPY") == 0)
      return "AUDJPY";
   if(StringFind(normalized, "NZDJPY") == 0)
      return "NZDJPY";
   if(StringFind(normalized, "AUDNZD") == 0)
      return "AUDNZD";
   if(StringFind(normalized, "EURJPY") == 0)
      return "EURJPY";
   if(StringFind(normalized, "GBPJPY") == 0)
      return "GBPJPY";
   if(StringFind(normalized, "AAPL") == 0)
      return "AAPL";
   if(StringFind(normalized, "NVIDIA") == 0 || StringFind(normalized, "NVDA") == 0)
      return "NVIDIA";
   return normalized;
}

bool SymbolsCompatible(string left, string right)
{
   return NormalizeBridgeSymbol(left) == NormalizeBridgeSymbol(right);
}

bool IsAlreadyFlatCloseReason(string reason)
{
   string upper = StringToUpper(TrimStr(reason));
   if(upper == "")
      return false;
   if(StringFind(upper, "POSITION DOESN'T EXIST") >= 0 || StringFind(upper, "POSITION DOES NOT EXIST") >= 0)
      return true;
   if(StringFind(upper, "ALREADY FLAT") >= 0 || StringFind(upper, "POSITION CLOSED") >= 0)
      return true;
   if(StringFind(upper, "UNKNOWN POSITION") >= 0)
      return true;
   return false;
}

int CountManagedOpenPositions(string symbol)
{
   int count = 0;
   string scopedSymbol = ResolveExecutionSymbol(symbol);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != magicNumber)
         continue;
      if(!SymbolsCompatible(PositionGetString(POSITION_SYMBOL), scopedSymbol))
         continue;
      count++;
   }
   return count;
}

string ResolveExecutionSymbol(string decisionSymbol)
{
   string trimmed = TrimStr(decisionSymbol);
   if(trimmed == "")
      return _Symbol;
   if(SymbolsCompatible(trimmed, _Symbol))
      return _Symbol;
   return trimmed;
}

int DateKey(datetime ts)
{
   MqlDateTime dt;
   TimeToStruct(ts, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}

void RefreshDayAnchor()
{
   int currentKey = DateKey(TimeCurrent());
   if(g_dayKey != currentKey)
   {
      g_dayKey = currentKey;
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   }
}

double DailyRealizedPnl()
{
   datetime dayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   if(!HistorySelect(dayStart, TimeCurrent()))
      return 0.0;
   double pnl = 0.0;
   int total = (int)HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong deal = HistoryDealGetTicket(i);
      if(deal == 0)
         continue;
      if((long)HistoryDealGetInteger(deal, DEAL_MAGIC) != magicNumber)
         continue;
      if((int)HistoryDealGetInteger(deal, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;
      pnl += HistoryDealGetDouble(deal, DEAL_PROFIT);
      pnl += HistoryDealGetDouble(deal, DEAL_SWAP);
      pnl += HistoryDealGetDouble(deal, DEAL_COMMISSION);
   }
   return pnl;
}

double DailyLossPct()
{
   if(g_dayStartEquity <= 0.0)
      return 0.0;
   double pnl = DailyRealizedPnl();
   if(pnl >= 0.0)
      return 0.0;
   return (-pnl / g_dayStartEquity) * 100.0;
}

int CountOpenPositions(string symbol = "")
{
   int count = 0;
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != magicNumber)
         continue;
      string posSymbol = PositionGetString(POSITION_SYMBOL);
      if(symbol != "" && posSymbol != symbol)
         continue;
      count++;
   }
   return count;
}

double CurrentSpreadPoints(string symbol)
{
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(point <= 0.0)
      return 0.0;
   return (ask - bid) / point;
}

bool CanOpenNewTradesForSymbol(string symbol, string &reason)
{
   reason = "";
   if(!g_backendReachable)
   {
      reason = "backend_unreachable";
      return false;
   }
   if(CountOpenPositions() >= maxPositions)
   {
      reason = "max_positions_reached";
      return false;
   }
   string scopedSymbol = (TrimStr(symbol) == "") ? _Symbol : symbol;
   if(CountOpenPositions(scopedSymbol) >= maxPerSymbol)
   {
      reason = "max_positions_symbol_reached";
      return false;
   }
   if(CurrentSpreadPoints(scopedSymbol) > spreadMaxPoints)
   {
      reason = "spread_too_wide";
      return false;
   }
   if(DailyLossPct() >= killSwitchDailyLoss)
   {
      reason = "daily_loss_killswitch";
      return false;
   }
   return true;
}

string JsonGetRaw(string json, string key)
{
   string token = "\"" + key + "\":";
   int start = StringFind(json, token);
   if(start < 0)
      return "";
   start += StringLen(token);
   int length = StringLen(json);
   while(start < length)
   {
      ushort c = StringGetCharacter(json, start);
      if(c == ' ' || c == '\t' || c == '\r' || c == '\n')
         start++;
      else
         break;
   }
   if(start >= length)
      return "";
   ushort first = StringGetCharacter(json, start);
   if(first == '\"')
   {
      int end = start + 1;
      while(end < length)
      {
         ushort cc = StringGetCharacter(json, end);
         ushort prev = StringGetCharacter(json, end - 1);
         if(cc == '\"' && prev != '\\')
            break;
         end++;
      }
      if(end >= length)
         return "";
      return StringSubstr(json, start + 1, end - start - 1);
   }
   if(first == '{' || first == '[')
   {
      ushort openChar = first;
      ushort closeChar = (first == '{') ? '}' : ']';
      int depth = 0;
      int endObj = start;
      for(; endObj < length; endObj++)
      {
         ushort cc = StringGetCharacter(json, endObj);
         if(cc == openChar)
            depth++;
         if(cc == closeChar)
         {
            depth--;
            if(depth == 0)
               break;
         }
      }
      if(endObj >= length)
         return "";
      return StringSubstr(json, start, endObj - start + 1);
   }
   int endNum = start;
   while(endNum < length)
   {
      ushort cc = StringGetCharacter(json, endNum);
      if(cc == ',' || cc == '}' || cc == ']')
         break;
      endNum++;
   }
   return TrimStr(StringSubstr(json, start, endNum - start));
}

bool JsonGetBool(string json, string key, bool defaultValue)
{
   string raw = StringToLower(TrimStr(JsonGetRaw(json, key)));
   if(raw == "true")
      return true;
   if(raw == "false")
      return false;
   return defaultValue;
}

double JsonGetDouble(string json, string key, double defaultValue)
{
   string raw = TrimStr(JsonGetRaw(json, key));
   if(raw == "")
      return defaultValue;
   return StringToDouble(raw);
}

int JsonGetInt(string json, string key, int defaultValue)
{
   string raw = TrimStr(JsonGetRaw(json, key));
   if(raw == "")
      return defaultValue;
   return (int)StringToInteger(raw);
}

string ExtractFirstAction(string payload)
{
   string actionsRaw = JsonGetRaw(payload, "actions");
   if(actionsRaw == "" || actionsRaw == "[]")
      return "";
   int objStart = StringFind(actionsRaw, "{");
   if(objStart < 0)
      return "";
   int depth = 0;
   int length = StringLen(actionsRaw);
   for(int i = objStart; i < length; i++)
   {
      ushort c = StringGetCharacter(actionsRaw, i);
      if(c == '{')
         depth++;
      if(c == '}')
      {
         depth--;
         if(depth == 0)
            return StringSubstr(actionsRaw, objStart, i - objStart + 1);
      }
   }
   return "";
}

string ExtractFirstObject(string arrayRaw)
{
   int objStart = StringFind(arrayRaw, "{");
   if(objStart < 0)
      return "";
   int depth = 0;
   int length = StringLen(arrayRaw);
   for(int i = objStart; i < length; i++)
   {
      ushort c = StringGetCharacter(arrayRaw, i);
      if(c == '{')
         depth++;
      if(c == '}')
      {
         depth--;
         if(depth == 0)
            return StringSubstr(arrayRaw, objStart, i - objStart + 1);
      }
   }
   return "";
}

bool ParseDecision(string actionRaw, DecisionData &decision)
{
   decision.signal_id = JsonGetRaw(actionRaw, "signal_id");
   decision.action = StringToUpper(JsonGetRaw(actionRaw, "action"));
   if(decision.action == "")
      decision.action = StringToUpper(JsonGetRaw(actionRaw, "action_type"));
   if(decision.action == "")
      decision.action = "OPEN_MARKET";
   decision.target_ticket = JsonGetRaw(actionRaw, "target_ticket");
   if(decision.target_ticket == "")
      decision.target_ticket = JsonGetRaw(actionRaw, "ticket");
   decision.symbol = JsonGetRaw(actionRaw, "symbol");
   decision.side = StringToUpper(JsonGetRaw(actionRaw, "side"));
   decision.lot = JsonGetDouble(actionRaw, "lot", 0.0);
   decision.sl = JsonGetDouble(actionRaw, "sl", 0.0);
   decision.tp = JsonGetDouble(actionRaw, "tp", 0.0);
   decision.max_slippage_points = JsonGetInt(actionRaw, "max_slippage_points", slippage);
   decision.reason = JsonGetRaw(actionRaw, "reason");
   decision.ai_summary = JsonGetRaw(actionRaw, "ai_summary");

   string trailingRaw = JsonGetRaw(actionRaw, "trailing");
   decision.trailing_enabled = JsonGetBool(trailingRaw, "enabled", true);
   decision.trailing_points = JsonGetInt(trailingRaw, "points", 250);
   decision.trailing_step_points = JsonGetInt(trailingRaw, "stepPoints", 50);

   string breakevenRaw = JsonGetRaw(actionRaw, "breakeven");
   decision.breakeven_enabled = JsonGetBool(breakevenRaw, "enabled", true);
   decision.breakeven_trigger_points = JsonGetInt(breakevenRaw, "triggerPoints", 120);

   string partialsRaw = JsonGetRaw(actionRaw, "partials");
   string partialRaw = ExtractFirstObject(partialsRaw);
   if(partialRaw == "")
   {
      decision.partial_rr = 0.0;
      decision.partial_percent = 0.0;
   }
   else
   {
      decision.partial_rr = 0.0;
      decision.partial_percent = 0.0;
   }

   if(decision.signal_id == "")
      return false;
   if(decision.symbol == "")
      decision.symbol = _Symbol;
   if(decision.action == "CLOSE_ALL")
      return true;
   if(decision.action == "CLOSE_POSITION")
      return (decision.target_ticket != "");
   if(decision.action == "MODIFY_SLTP")
   {
      if(decision.target_ticket == "")
         return false;
      if(decision.sl <= 0.0 && decision.tp <= 0.0)
         return false;
      return true;
   }
   if(decision.action != "OPEN_MARKET")
      return false;
   if(decision.side != "BUY" && decision.side != "SELL")
      return false;
   if(decision.lot <= 0.0)
      return false;
   return true;
}

bool HttpJson(string method, string endpoint, string body, string &response, int &status)
{
   string url = serverUrl + endpoint;
   string headers = "Content-Type: application/json\r\n";
   if(apiKey != "")
      headers = headers + "X-Bridge-Token: " + apiKey + "\r\n";

   char data[];
   if(method == "POST")
      StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   else
      ArrayResize(data, 0);

   char result[];
   string resultHeaders;
   ResetLastError();
   int code = WebRequest(method, url, headers, 8000, data, result, resultHeaders);
   if(code == -1)
   {
      status = -1;
      response = "";
      int errorCode = GetLastError();
      g_lastErrorReason = StringFormat("WebRequest error=%d endpoint=%s", errorCode, endpoint);
      LogLine("WARN", g_lastErrorReason);
      return false;
   }
   status = code;
   response = CharArrayToString(result, 0, -1, CP_UTF8);
   return true;
}

bool SendJsonExpectOk(string endpoint, string body, string &response, int &status)
{
   if(!HttpJson("POST", endpoint, body, response, status))
      return false;
   return (status >= 200 && status < 300);
}

bool DispatchReliableReport(string key, string endpoint, string payload)
{
   string response;
   int status = 0;
   if(SendJsonExpectOk(endpoint, payload, response, status))
   {
      int existingIndex = FindPendingReportIndex(key);
      if(existingIndex >= 0)
         RemovePendingReportAt(existingIndex);
      return true;
   }

   int attempts = 0;
   int existingIndex = FindPendingReportIndex(key);
   if(existingIndex >= 0)
      attempts = g_pendingReports[existingIndex].attempts;
   QueuePendingReport(key, endpoint, payload, attempts + 1);
   LogLine("WARN", StringFormat("Pending report queued key=%s endpoint=%s status=%d", key, endpoint, status));
   return false;
}

void FlushPendingReports(int limit = 8)
{
   int flushed = 0;
   datetime now = TimeCurrent();
   for(int i = 0; i < ArraySize(g_pendingReports) && flushed < limit;)
   {
      if(g_pendingReports[i].next_attempt > now)
      {
         i++;
         continue;
      }
      string response;
      int status = 0;
      if(SendJsonExpectOk(g_pendingReports[i].endpoint, g_pendingReports[i].payload, response, status))
      {
         LogLine("INFO", StringFormat("Pending report flushed key=%s endpoint=%s", g_pendingReports[i].key, g_pendingReports[i].endpoint));
         RemovePendingReportAt(i);
         flushed++;
         continue;
      }
      g_pendingReports[i].attempts = MathMax(1, g_pendingReports[i].attempts + 1);
      g_pendingReports[i].next_attempt = TimeCurrent() + MathMin(60, MathMax(2, g_pendingReports[i].attempts * 5));
      SavePendingReports();
      LogLine("WARN", StringFormat("Pending report retry deferred key=%s endpoint=%s status=%d", g_pendingReports[i].key, g_pendingReports[i].endpoint, status));
      i++;
      flushed++;
   }
}

void ReportExecution(string signalId, bool accepted, ulong ticket, double price, double slipPoints, int retcode, string reason, string symbol = "", string side = "", double lot = 0.0)
{
   string payload = StringFormat(
      "{\"signal_id\":\"%s\",\"accepted\":%s,\"ticket\":\"%I64u\",\"entry_price\":%.8f,\"slippage_points\":%.2f,\"retcode\":%d,\"reason\":\"%s\",\"symbol\":\"%s\",\"side\":\"%s\",\"lot\":%.2f,\"account\":\"%I64d\",\"magic\":%I64d,\"equity\":%.2f}",
      EscapeJson(signalId),
      accepted ? "true" : "false",
      ticket,
      price,
      slipPoints,
      retcode,
      EscapeJson(reason),
      EscapeJson(symbol),
      EscapeJson(side),
      lot,
      AccountInfoInteger(ACCOUNT_LOGIN),
      magicNumber,
      AccountInfoDouble(ACCOUNT_EQUITY)
   );
   string key = StringFormat("exec:%s:%I64u", signalId, ticket);
   DispatchReliableReport(key, "/v1/report_execution", payload);
}

void ReportClose(string signalId, double exitPrice, double pnlMoney, string reason, string symbol = "")
{
   string payload = StringFormat(
      "{\"signal_id\":\"%s\",\"exit_price\":%.8f,\"pnl_money\":%.2f,\"reason\":\"%s\",\"symbol\":\"%s\",\"account\":\"%I64d\",\"magic\":%I64d,\"equity_after_close\":%.2f,\"closed_at\":\"%s\"}",
      EscapeJson(signalId),
      exitPrice,
      pnlMoney,
      EscapeJson(reason),
      EscapeJson(symbol),
      AccountInfoInteger(ACCOUNT_LOGIN),
      magicNumber,
      AccountInfoDouble(ACCOUNT_EQUITY),
      TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS)
   );
   string key = StringFormat("close:%s", signalId);
   DispatchReliableReport(key, "/v1/report_close", payload);
}

void SendHeartbeat()
{
   string payload = StringFormat(
      "{\"account\":%I64d,\"symbol\":\"%s\",\"magic\":%I64d,\"open_positions\":%d,\"backend_reachable\":%s,\"last_execution\":\"%s\"}",
      AccountInfoInteger(ACCOUNT_LOGIN),
      _Symbol,
      magicNumber,
      CountOpenPositions(),
      g_backendReachable ? "true" : "false",
      EscapeJson(g_lastExecutionSummary)
   );
   string response;
   int status = 0;
   HttpJson("POST", "/v1/heartbeat", payload, response, status);
}

int FindRuleIndex(ulong ticket)
{
   int total = ArraySize(g_rules);
   for(int i = 0; i < total; i++)
   {
      if(g_rules[i].ticket == ticket)
         return i;
   }
   return -1;
}

void TrackRule(ulong ticket, DecisionData &decision)
{
   if(ticket == 0)
      return;
   int index = FindRuleIndex(ticket);
   if(index < 0)
   {
      index = ArraySize(g_rules);
      ArrayResize(g_rules, index + 1);
   }
   g_rules[index].ticket = ticket;
   g_rules[index].signal_id = decision.signal_id;
   g_rules[index].trailing_enabled = decision.trailing_enabled;
   g_rules[index].trailing_points = MathMax(50, decision.trailing_points);
   g_rules[index].trailing_step_points = MathMax(10, decision.trailing_step_points);
   g_rules[index].breakeven_enabled = decision.breakeven_enabled;
   g_rules[index].breakeven_trigger_points = MathMax(10, decision.breakeven_trigger_points);
   g_rules[index].breakeven_done = false;
   g_rules[index].partial_done = false;
   g_rules[index].partial_rr = decision.partial_rr;
   g_rules[index].partial_percent = decision.partial_percent;
   g_rules[index].initial_sl = decision.sl;
}

ulong ResolvePositionTicket(string executionSymbol, string signalId, ulong fallbackTicket)
{
   if(fallbackTicket > 0 && PositionSelectByTicket(fallbackTicket))
      return fallbackTicket;

   ulong bestTicket = fallbackTicket;
   long bestOpenedAt = -1;
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != magicNumber)
         continue;
      if(!SymbolsCompatible(PositionGetString(POSITION_SYMBOL), executionSymbol))
         continue;
      string comment = PositionGetString(POSITION_COMMENT);
      long openedAt = (long)PositionGetInteger(POSITION_TIME_MSC);
      if(comment == signalId)
         return ticket;
      if(openedAt >= bestOpenedAt)
      {
         bestOpenedAt = openedAt;
         bestTicket = ticket;
      }
   }
   return bestTicket;
}

void CleanupRules()
{
   int total = ArraySize(g_rules);
   for(int i = total - 1; i >= 0; i--)
   {
      if(!PositionSelectByTicket(g_rules[i].ticket))
      {
         for(int j = i; j < ArraySize(g_rules) - 1; j++)
            g_rules[j] = g_rules[j + 1];
         ArrayResize(g_rules, ArraySize(g_rules) - 1);
      }
   }
}

void ManageOpenPositions()
{
   int total = PositionsTotal();
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.0001;

   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != magicNumber)
         continue;
      string symbol = PositionGetString(POSITION_SYMBOL);
      if(symbol != _Symbol)
         continue;

      int ruleIndex = FindRuleIndex(ticket);
      if(ruleIndex < 0)
         continue;
      PositionRule rule = g_rules[ruleIndex];

      long type = PositionGetInteger(POSITION_TYPE);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      double current = (type == POSITION_TYPE_BUY) ? bid : ask;
      double profitPoints = (type == POSITION_TYPE_BUY) ? (current - entry) / point : (entry - current) / point;
      double riskPoints = MathAbs(entry - rule.initial_sl) / point;

      if(rule.breakeven_enabled && !rule.breakeven_done && profitPoints >= rule.breakeven_trigger_points)
      {
         double newSl = entry + ((type == POSITION_TYPE_BUY) ? point : -point);
         if(trade.PositionModify(ticket, newSl, tp))
         {
            g_rules[ruleIndex].breakeven_done = true;
            LogLine("INFO", StringFormat("BE moved ticket=%I64u signal=%s", ticket, rule.signal_id));
         }
      }

      if(rule.trailing_enabled && profitPoints >= rule.trailing_points)
      {
         double newSl = (type == POSITION_TYPE_BUY) ? current - (rule.trailing_points * point) : current + (rule.trailing_points * point);
         bool shouldMove = false;
         if(type == POSITION_TYPE_BUY)
            shouldMove = (sl <= 0.0 || newSl > (sl + (rule.trailing_step_points * point)));
         else
            shouldMove = (sl <= 0.0 || newSl < (sl - (rule.trailing_step_points * point)));
         if(shouldMove && trade.PositionModify(ticket, newSl, tp))
            LogLine("INFO", StringFormat("Trail updated ticket=%I64u signal=%s", ticket, rule.signal_id));
      }

      if(!rule.partial_done && rule.partial_percent > 0.0 && rule.partial_rr > 0.0 && riskPoints > 0.0)
      {
         double triggerPoints = riskPoints * rule.partial_rr;
         if(profitPoints >= triggerPoints)
         {
            double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
            double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
            double closeVolume = volume * (rule.partial_percent / 100.0);
            closeVolume = MathMax(minLot, MathFloor(closeVolume / lotStep) * lotStep);
            if(closeVolume > 0.0 && closeVolume < volume)
            {
               if(trade.PositionClosePartial(ticket, closeVolume, (ulong)slippage))
               {
                  g_rules[ruleIndex].partial_done = true;
                  LogLine("INFO", StringFormat("Partial close ticket=%I64u signal=%s volume=%.2f", ticket, rule.signal_id, closeVolume));
               }
            }
         }
      }
   }
   CleanupRules();
}

void ScanAndReportClosedDeals()
{
   datetime fromTs = TimeCurrent() - 86400;
   if(!HistorySelect(fromTs, TimeCurrent()))
      return;
   int total = (int)HistoryDealsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong deal = HistoryDealGetTicket(i);
      if(deal == 0)
         continue;
      string closeId = IntegerToString((long)deal);
      if(InStringArray(g_reportedDealIds, closeId))
         continue;
      if((long)HistoryDealGetInteger(deal, DEAL_MAGIC) != magicNumber)
         continue;
      if((int)HistoryDealGetInteger(deal, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;
      string symbol = HistoryDealGetString(deal, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;

      string signalId = HistoryDealGetString(deal, DEAL_COMMENT);
      if(signalId == "")
         continue;
      double exitPrice = HistoryDealGetDouble(deal, DEAL_PRICE);
      double pnl = HistoryDealGetDouble(deal, DEAL_PROFIT) + HistoryDealGetDouble(deal, DEAL_SWAP) + HistoryDealGetDouble(deal, DEAL_COMMISSION);
      ReportClose(signalId, exitPrice, pnl, "deal_closed", symbol);
      PushUnique(g_reportedDealIds, closeId);
   }
}

void ExecuteDecision(DecisionData &decision)
{
   if(!SymbolsCompatible(decision.symbol, _Symbol))
   {
      LogLine("WARN", StringFormat("symbol_mismatch signal=%s decision_symbol=%s chart_symbol=%s", decision.signal_id, decision.symbol, _Symbol));
      ReportExecution(decision.signal_id, false, 0, 0.0, 0.0, -1, "symbol_mismatch", decision.symbol, decision.side, decision.lot);
      return;
   }
   string executionSymbol = ResolveExecutionSymbol(decision.symbol);
   if(IsExecutedSignal(decision.signal_id))
   {
      LogLine("INFO", "Duplicate signal ignored (already executed): " + decision.signal_id);
      ReportExecution(decision.signal_id, false, 0, 0.0, 0.0, -1, "duplicate_signal_already_executed", executionSymbol, decision.side, decision.lot);
      return;
   }
   if(g_lastDecisionId == decision.signal_id && (TimeCurrent() - g_lastDecisionAt) < duplicateWindowSeconds)
   {
      LogLine("INFO", "Duplicate signal ignored (time window): " + decision.signal_id);
      ReportExecution(decision.signal_id, false, 0, 0.0, 0.0, -1, "duplicate_signal_window", executionSymbol, decision.side, decision.lot);
      return;
   }
   trade.SetExpertMagicNumber(magicNumber);
   trade.SetDeviationInPoints((ulong)MathMax(1, MathMin(decision.max_slippage_points, slippage)));

   if(decision.action == "CLOSE_ALL")
   {
      int closed = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0 || !PositionSelectByTicket(ticket))
            continue;
         if((long)PositionGetInteger(POSITION_MAGIC) != magicNumber)
            continue;
         if(!SymbolsCompatible(PositionGetString(POSITION_SYMBOL), executionSymbol))
            continue;
         if(trade.PositionClose(ticket, (ulong)slippage))
            closed++;
      }
      bool accepted = (closed > 0 || CountOpenPositions(decision.symbol) == 0);
      RememberExecutedSignal(decision.signal_id);
      g_lastDecisionId = decision.signal_id;
      g_lastDecisionAt = TimeCurrent();
      g_lastExecutionSummary = StringFormat("action=%s signal=%s closed=%d", decision.action, decision.signal_id, closed);
      LogLine(accepted ? "INFO" : "WARN", g_lastExecutionSummary + " reason=" + decision.reason);
      ReportExecution(decision.signal_id, accepted, 0, 0.0, 0.0, accepted ? 0 : -1, accepted ? "close_all_done" : "close_all_failed", executionSymbol, decision.side, decision.lot);
      return;
   }

   if(decision.action == "CLOSE_POSITION")
   {
      ulong targetTicket = (ulong)StringToInteger(decision.target_ticket);
      bool accepted = false;
      int closeRetcode = -1;
      string closeReason = "close_position_failed";
      if(targetTicket > 0 && PositionSelectByTicket(targetTicket))
      {
         if((long)PositionGetInteger(POSITION_MAGIC) == magicNumber && SymbolsCompatible(PositionGetString(POSITION_SYMBOL), executionSymbol))
         {
            accepted = trade.PositionClose(targetTicket, (ulong)slippage);
            closeRetcode = (int)trade.ResultRetcode();
            if(accepted)
               closeReason = "close_position_done";
            else if(IsAlreadyFlatCloseReason(RetcodeToString(trade.ResultRetcode())))
            {
               accepted = true;
               closeRetcode = 0;
               closeReason = "close_position_already_flat";
            }
         }
      }
      else if(targetTicket > 0 && CountManagedOpenPositions(executionSymbol) == 0)
      {
         accepted = true;
         closeRetcode = 0;
         closeReason = "close_position_already_flat";
      }
      RememberExecutedSignal(decision.signal_id);
      g_lastDecisionId = decision.signal_id;
      g_lastDecisionAt = TimeCurrent();
      g_lastExecutionSummary = StringFormat("action=%s signal=%s ticket=%I64u accepted=%s reason=%s", decision.action, decision.signal_id, targetTicket, accepted ? "true" : "false", closeReason);
      LogLine(accepted ? "INFO" : "WARN", g_lastExecutionSummary);
      ReportExecution(decision.signal_id, accepted, targetTicket, 0.0, 0.0, accepted ? 0 : closeRetcode, closeReason, executionSymbol, decision.side, decision.lot);
      return;
   }

   if(decision.action == "MODIFY_SLTP")
   {
      ulong targetTicket = (ulong)StringToInteger(decision.target_ticket);
      bool accepted = false;
      double currentBid = SymbolInfoDouble(executionSymbol, SYMBOL_BID);
      double currentAsk = SymbolInfoDouble(executionSymbol, SYMBOL_ASK);
      int digits = (int)SymbolInfoInteger(executionSymbol, SYMBOL_DIGITS);
      double currentTp = 0.0;
      double currentSl = 0.0;
      double nextSl = decision.sl;
      double nextTp = decision.tp;
      if(targetTicket > 0 && PositionSelectByTicket(targetTicket))
      {
         if((long)PositionGetInteger(POSITION_MAGIC) == magicNumber && SymbolsCompatible(PositionGetString(POSITION_SYMBOL), executionSymbol))
         {
            currentTp = PositionGetDouble(POSITION_TP);
            currentSl = PositionGetDouble(POSITION_SL);
            nextSl = (decision.sl > 0.0) ? decision.sl : currentSl;
            nextTp = (decision.tp > 0.0) ? decision.tp : currentTp;
            accepted = trade.PositionModify(targetTicket, nextSl, nextTp);
         }
      }
      RememberExecutedSignal(decision.signal_id);
      g_lastDecisionId = decision.signal_id;
      g_lastDecisionAt = TimeCurrent();
      g_lastExecutionSummary = StringFormat("action=%s signal=%s ticket=%I64u accepted=%s", decision.action, decision.signal_id, targetTicket, accepted ? "true" : "false");
      if(!accepted)
      {
         LogLine(
            "WARN",
            g_lastExecutionSummary
            + StringFormat(
               " retcode=%d(%s) requested_sl=%s requested_tp=%s current_sl=%s current_tp=%s bid=%s ask=%s",
               (int)trade.ResultRetcode(),
               RetcodeToString(trade.ResultRetcode()),
               DoubleToString(nextSl, digits),
               DoubleToString(nextTp, digits),
               DoubleToString(currentSl, digits),
               DoubleToString(currentTp, digits),
               DoubleToString(currentBid, digits),
               DoubleToString(currentAsk, digits)
            )
         );
      }
      else
      {
         LogLine("INFO", g_lastExecutionSummary);
      }
      ReportExecution(decision.signal_id, accepted, targetTicket, 0.0, 0.0, accepted ? 0 : -1, accepted ? "modify_done" : "modify_failed", executionSymbol, decision.side, decision.lot);
      return;
   }

   string guardReason = "";
   if(!CanOpenNewTradesForSymbol(executionSymbol, guardReason))
   {
      LogLine("WARN", "open_market_blocked_by_local_guards signal=" + decision.signal_id + " reason=" + guardReason);
      ReportExecution(decision.signal_id, false, 0, 0.0, 0.0, -1, guardReason, executionSymbol, decision.side, decision.lot);
      return;
   }

   bool ok = false;
   if(decision.side == "BUY")
      ok = trade.Buy(decision.lot, executionSymbol, 0.0, decision.sl, decision.tp, decision.signal_id);
   else
      ok = trade.Sell(decision.lot, executionSymbol, 0.0, decision.sl, decision.tp, decision.signal_id);

   int retcode = (int)trade.ResultRetcode();
   ulong orderTicket = trade.ResultOrder();
   ulong dealTicket = trade.ResultDeal();
   double fillPrice = trade.ResultPrice();
   double currentBid = SymbolInfoDouble(executionSymbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(executionSymbol, SYMBOL_ASK);
   int digits = (int)SymbolInfoInteger(executionSymbol, SYMBOL_DIGITS);
   int stopsLevel = (int)SymbolInfoInteger(executionSymbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freezeLevel = (int)SymbolInfoInteger(executionSymbol, SYMBOL_TRADE_FREEZE_LEVEL);
   double marketPrice = (decision.side == "BUY") ? currentAsk : currentBid;
   double point = SymbolInfoDouble(executionSymbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.0001;
   double slipPoints = MathAbs(fillPrice - marketPrice) / point;

   if(ok)
   {
      ulong positionTicket = ResolvePositionTicket(executionSymbol, decision.signal_id, orderTicket);
      RememberExecutedSignal(decision.signal_id);
      g_lastDecisionId = decision.signal_id;
      g_lastDecisionAt = TimeCurrent();
      g_lastExecutionSummary = StringFormat("accepted signal=%s order=%I64u deal=%I64u position=%I64u", decision.signal_id, orderTicket, dealTicket, positionTicket);
      TrackRule(positionTicket, decision);
      LogLine("INFO", g_lastExecutionSummary + " reason=" + decision.reason);
      ReportExecution(decision.signal_id, true, positionTicket, fillPrice, slipPoints, retcode, "accepted", executionSymbol, decision.side, decision.lot);
      return;
   }

   string reason = StringFormat("rejected retcode=%d", retcode);
   g_lastExecutionSummary = StringFormat("rejected signal=%s %s", decision.signal_id, reason);
   LogLine(
      "WARN",
      g_lastExecutionSummary
      + StringFormat(
         " side=%s symbol=%s requested_sl=%s requested_tp=%s bid=%s ask=%s stops_level=%d freeze_level=%d retcode=%d(%s)",
         decision.side,
         executionSymbol,
         DoubleToString(decision.sl, digits),
         DoubleToString(decision.tp, digits),
         DoubleToString(currentBid, digits),
         DoubleToString(currentAsk, digits),
         stopsLevel,
         freezeLevel,
         retcode,
         RetcodeToString(retcode)
      )
   );
   ReportExecution(decision.signal_id, false, orderTicket, fillPrice, slipPoints, retcode, reason, executionSymbol, decision.side, decision.lot);
}

void PullAndExecute()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double last = SymbolInfoDouble(_Symbol, SYMBOL_LAST);
   if(last <= 0.0 && bid > 0.0 && ask > 0.0)
      last = (bid + ask) * 0.5;
   else if(last <= 0.0)
      last = MathMax(bid, ask);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(point <= 0.0)
      point = 0.0001;
   double spreadPoints = (bid > 0.0 && ask > 0.0) ? MathAbs(ask - bid) / point : 0.0;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lotMin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotMax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double contractSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   int stopsLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freezeLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   int symbolSelected = (int)SymbolInfoInteger(_Symbol, SYMBOL_SELECT);
   int symbolTradeMode = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);
   int terminalConnected = (int)TerminalInfoInteger(TERMINAL_CONNECTED);
   int terminalTradeAllowed = (int)TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   int mqlTradeAllowed = (int)MQLInfoInteger(MQL_TRADE_ALLOWED);
   long serverTime = (long)TimeTradeServer();
   long gmtTime = (long)TimeGMT();
   string endpoint = StringFormat(
      "/v1/pull?symbol=%s&account=%I64d&magic=%I64d&balance=%.2f&equity=%.2f&free_margin=%.2f"
      + "&bid=%.10f&ask=%.10f&last=%.10f&spread_points=%.2f&point=%.10f&digits=%d"
      + "&tick_size=%.10f&tick_value=%.10f&lot_min=%.4f&lot_max=%.4f&lot_step=%.4f&contract_size=%.4f"
      + "&stops_level=%d&freeze_level=%d&symbol_selected=%d&symbol_trade_mode=%d"
      + "&terminal_connected=%d&terminal_trade_allowed=%d&mql_trade_allowed=%d&server_time=%I64d&gmt_time=%I64d",
      _Symbol,
      AccountInfoInteger(ACCOUNT_LOGIN),
      magicNumber,
      AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoDouble(ACCOUNT_EQUITY),
      AccountInfoDouble(ACCOUNT_MARGIN_FREE),
      bid,
      ask,
      last,
      spreadPoints,
      point,
      digits,
      tickSize,
      tickValue,
      lotMin,
      lotMax,
      lotStep,
      contractSize,
      stopsLevel,
      freezeLevel,
      symbolSelected,
      symbolTradeMode,
      terminalConnected,
      terminalTradeAllowed,
      mqlTradeAllowed,
      serverTime,
      gmtTime
   );
   string response;
   int status = 0;
   if(!HttpJson("GET", endpoint, "", response, status))
   {
      g_backendReachable = false;
      return;
   }
   if(status != 200)
   {
      g_backendReachable = false;
      g_lastErrorReason = StringFormat("pull_http_status=%d", status);
      LogLine("WARN", g_lastErrorReason);
      return;
   }
   g_backendReachable = true;
   string actionRaw = ExtractFirstAction(response);
   if(actionRaw == "")
      return;
   DecisionData decision;
   if(!ParseDecision(actionRaw, decision))
   {
      LogLine("WARN", "Invalid action payload: " + actionRaw);
      string invalidSignalId = JsonGetRaw(actionRaw, "signal_id");
      if(invalidSignalId != "")
         ReportExecution(
            invalidSignalId,
            false,
            0,
            0.0,
            0.0,
            -1,
            "invalid_action_payload",
            JsonGetRaw(actionRaw, "symbol"),
            StringToUpper(JsonGetRaw(actionRaw, "side")),
            JsonGetDouble(actionRaw, "lot", 0.0)
         );
      return;
   }
   ExecuteDecision(decision);
}

int OnInit()
{
   trade.SetExpertMagicNumber(magicNumber);
   EventSetTimer(1);
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayKey = DateKey(TimeCurrent());
   LoadExecutedSignals();
   LoadPendingReports();
   ArrayResize(g_reportedDealIds, 0);
   ArrayResize(g_rules, 0);
   LogLine("INFO", StringFormat("Init server=%s poll=%d", serverUrl, pollSeconds));
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   LogLine("INFO", StringFormat("Deinit reason=%d", reason));
}

void OnTick()
{
   // Polling and management run in OnTimer to avoid tick-burst over-calling WebRequest.
}

void OnTimer()
{
   RefreshDayAnchor();
   FlushPendingReports();
   ManageOpenPositions();
   ScanAndReportClosedDeals();

   if((TimeCurrent() - g_lastHeartbeat) >= 15)
   {
      g_lastHeartbeat = TimeCurrent();
      SendHeartbeat();
   }

   if((TimeCurrent() - g_lastPoll) < pollSeconds)
      return;
   g_lastPoll = TimeCurrent();
   PullAndExecute();
}
