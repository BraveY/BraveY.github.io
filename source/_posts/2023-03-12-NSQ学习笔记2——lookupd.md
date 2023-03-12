---
title: NSQ学习笔记2——lookupd
date: 2023-03-12 10:14:41
categories: NSQ
tags:
- Go
- 消息队列
copyright: true
---
在官方教程Demo中，首先需要启动的程序就是lookupd，因此首先从lookupd程序进行学习。loookupd的作用参见上篇[简介文章](https://bravey.github.io/2023-02-05-NSQ%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B01%E2%80%94%E2%80%94%E7%AE%80%E4%BB%8B.html)，作为中间件将生产者与消费者解耦，通过lookupd进行节点发现，以及生产者和消费者的互相发现。

# 启动

main函数的位置在 `nsq/apps/nsqlookupd/main.go`文件中。代码很简洁：

```go
func main() {
	prg := &program{}
	if err := svc.Run(prg, syscall.SIGINT, syscall.SIGTERM); err != nil {
		logFatal("%s", err)
	}
}
```
这里面主要使用svc这个运行框架包来提供程序的启动和安全退出。使用svc这个启动框架监听SIGINT和SIGTERM这两个信号，通过安全退出来保证相关defer函数得以执行，以及关键日志能够被记录。如果不监听信号，kill -9这样的强制退出会使得程序的defer函数等不执行，没有退出日志也会使得后续排查问题比较困难。

svc运行框架支持linux和windows，使用的时候需要实现Service的Init(),Start(),Stop()三个接口就可以了
```go
type Service interface {
	Init(Environment) error
	Start() error
	Stop() error
}
```
lookupd的Start()初始了配置，并在New()函数中启动两个TCP端口监听，随后指向了真正的Main()启动逻辑。
```go
func (p *program) Start() error {
   //...
	nsqlookupd, err := nsqlookupd.New(opts)
	if err != nil {
		logFatal("failed to instantiate nsqlookupd", err)
	}
	p.nsqlookupd = nsqlookupd

	go func() {
		err := p.nsqlookupd.Main()
		if err != nil {
			p.Stop()
			os.Exit(1)
		}
	}()

	return nil
}
```
在New()中进行最核心的用于保存生产者和消费者关系这一管控元数据的DB数据结构的初始化，并且启动的两个TCP端口监听，分别对应后面的TCP服务和HTTP协议的服务器。
```go
func New(opts *Options) (*NSQLookupd, error) {
   //...
	l := &NSQLookupd{
		opts: opts,
		DB:   NewRegistrationDB(), // 核心的元数据信息保存
	}
	l.tcpServer = &tcpServer{nsqlookupd: l}
	l.tcpListener, err = net.Listen("tcp", opts.TCPAddress) // TCP协议监听
	if err != nil {
		return nil, fmt.Errorf("listen (%s) failed - %s", opts.TCPAddress, err)
	}
	l.httpListener, err = net.Listen("tcp", opts.HTTPAddress) // HTTP协议端口监听
	if err != nil {
		return nil, fmt.Errorf("listen (%s) failed - %s", opts.HTTPAddress, err)
	}

	return l, nil
}
```
Main()主要是指定退出通道并开启一个TCP服务器，和一个HTTPS服务器。
```go
func (l *NSQLookupd) Main() error {
	exitCh := make(chan error)
	var once sync.Once
	exitFunc := func(err error) {
		once.Do(func() {
			if err != nil {
				l.logf(LOG_FATAL, "%s", err)
			}
			exitCh <- err
		})
	}

	l.waitGroup.Wrap(func() {
		exitFunc(protocol.TCPServer(l.tcpListener, l.tcpServer, l.logf))
	})
	httpServer := newHTTPServer(l)
	l.waitGroup.Wrap(func() {
		exitFunc(http_api.Serve(l.httpListener, httpServer, "HTTP", l.logf))
	})

	err := <-exitCh
	return err
}
```

# 元数据结构体
RegistrationDB是lookupd的核心元数据保存结构体。lookupd的作用就是对这个结构体维护，根据相关的TCP和HTTP接口来对这个DB进行增删改查。
```go
type RegistrationDB struct {
	sync.RWMutex
	registrationMap map[Registration]ProducerMap
}

type Registration struct {
	Category string
	Key      string
	SubKey   string
}
type Registrations []Registration

type PeerInfo struct {
	lastUpdate       int64
	id               string
	RemoteAddress    string `json:"remote_address"`
	Hostname         string `json:"hostname"`
	BroadcastAddress string `json:"broadcast_address"`
	TCPPort          int    `json:"tcp_port"`
	HTTPPort         int    `json:"http_port"`
	Version          string `json:"version"`
}

type Producer struct {
	peerInfo     *PeerInfo
	tombstoned   bool
	tombstonedAt time.Time
}

type Producers []*Producer
type ProducerMap map[string]*Producer
```
RegistrationDB是一个map结构，也就是一个字典。这个map的key是Registration结构体类型。Category可以是"topic","channel","client"等，而Key则是具体topic名字，SubKey则是channel的名字。

RegistrationDB字典的值是ProducerMap类型。这个ProducerMap也是一个map结构，其值是生产者元数据的列表。生产者的元数据具体保存在PeerInfo结构体中，包含端口，地址，心跳时间等
# TCP Server
TCP Server是一个死循环。套接字编程的实现，先listen()对端口监听，然后accept()来具体处理。具体而言每当Accpet()建立一个TCP连接后就会启动一个goroutine去进行处理，同时用WaitGroup来等待所有的goroutine处理完毕后再关闭服务器。
```go
func TCPServer(listener net.Listener, handler TCPHandler, logf lg.AppLogFunc) error {
	logf(lg.INFO, "TCP: listening on %s", listener.Addr())

	var wg sync.WaitGroup

	for {
		clientConn, err := listener.Accept()
      // err 处理忽略
		wg.Add(1)
		go func() {
			handler.Handle(clientConn)
			wg.Done()
		}()
	}

	// wait to return until all handler goroutines complete
	wg.Wait()

	logf(lg.INFO, "TCP: closing %s", listener.Addr())

	return nil
}
```
具体到处理函数，处理函数的核心逻辑在IOLoop这个函数中，IOLoop是个死循环，所以除非有错误产生不然就一直读取，因此是长链接。
```go
func (p *LookupProtocolV1) IOLoop(c protocol.Client) error {
	var err error
	var line string

	client := c.(*ClientV1)

	reader := bufio.NewReader(client)
	for {
		line, err = reader.ReadString('\n')
		if err != nil {
			break
		}

		line = strings.TrimSpace(line)
		params := strings.Split(line, " ")

		var response []byte
		response, err = p.Exec(client, reader, params) //具体处理逻辑
      //...错误处理
	}

}
```

Exec中是TCP的Server提供的API包括PING,IDENTIFY,REGISTER,UNREGISTER四个。
```go
func (p *LookupProtocolV1) Exec(client *ClientV1, reader *bufio.Reader, params []string) ([]byte, error) {
	switch params[0] {
	case "PING":
		return p.PING(client, params)
	case "IDENTIFY":
		return p.IDENTIFY(client, reader, params[1:])
	case "REGISTER":
		return p.REGISTER(client, reader, params[1:])
	case "UNREGISTER":
		return p.UNREGISTER(client, reader, params[1:])
	}
	return nil, protocol.NewFatalClientErr(nil, "E_INVALID", fmt.Sprintf("invalid command %s", params[0]))
}
```
nsqd先往nsqlookupd identify获取peer info也就是客户端信息，然后去注册。注册的时候根据参数如果有channel，那么对应channel注册一个生产者，而topic是必定会注册对应生产者的，也就是说参数必须包含topic用来注册。识别完成后用ping来上报心跳。所有的TCP连接中都需要先执行IDENTIFY操作。
# HTTP Server
HTTP提供的API如下,核心的主要是topic，channel等的寻找。
```go
	router.Handle("GET", "/ping", http_api.Decorate(s.pingHandler, log, http_api.PlainText))
	router.Handle("GET", "/info", http_api.Decorate(s.doInfo, log, http_api.V1))

	// v1 negotiate
	router.Handle("GET", "/debug", http_api.Decorate(s.doDebug, log, http_api.V1))
	router.Handle("GET", "/lookup", http_api.Decorate(s.doLookup, log, http_api.V1))
	router.Handle("GET", "/topics", http_api.Decorate(s.doTopics, log, http_api.V1))
	router.Handle("GET", "/channels", http_api.Decorate(s.doChannels, log, http_api.V1))
	router.Handle("GET", "/nodes", http_api.Decorate(s.doNodes, log, http_api.V1))

	// only v1
	router.Handle("POST", "/topic/create", http_api.Decorate(s.doCreateTopic, log, http_api.V1))
	router.Handle("POST", "/topic/delete", http_api.Decorate(s.doDeleteTopic, log, http_api.V1))
	router.Handle("POST", "/channel/create", http_api.Decorate(s.doCreateChannel, log, http_api.V1))
	router.Handle("POST", "/channel/delete", http_api.Decorate(s.doDeleteChannel, log, http_api.V1))
	router.Handle("POST", "/topic/tombstone", http_api.Decorate(s.doTombstoneTopicProducer, log, http_api.V1))

	// debug
	router.HandlerFunc("GET", "/debug/pprof", pprof.Index)
	router.HandlerFunc("GET", "/debug/pprof/cmdline", pprof.Cmdline)
	router.HandlerFunc("GET", "/debug/pprof/symbol", pprof.Symbol)
	router.HandlerFunc("POST", "/debug/pprof/symbol", pprof.Symbol)
	router.HandlerFunc("GET", "/debug/pprof/profile", pprof.Profile)
	router.Handler("GET", "/debug/pprof/heap", pprof.Handler("heap"))
	router.Handler("GET", "/debug/pprof/goroutine", pprof.Handler("goroutine"))
	router.Handler("GET", "/debug/pprof/block", pprof.Handler("block"))
	router.Handler("GET", "/debug/pprof/threadcreate", pprof.Handler("threadcreate"))
```
这里有个值得借鉴的是Decorate()这个装饰器函数的写法，使用函数编程的方法，简洁的实现了装饰器模式，从而可以添加指定的功能。
```go
func Decorate(f APIHandler, ds ...Decorator) httprouter.Handle {
	decorated := f
	for _, decorate := range ds {
		decorated = decorate(decorated)
	}
	return func(w http.ResponseWriter, req *http.Request, ps httprouter.Params) {
		decorated(w, req, ps)
	}
}

```
Decorate的输入输出都是函数，这个就是函数编程的思路。

# 总结
1. lookupd核心功能就是对生产者，消费者元数据的维护。
2. 安全退出包括程序的安全退出，这个通过监听信号实现。另一个是服务的安全退出，使用contex来实现。
3. 函数编程写法的装饰器函数。


# 参考资料

[服务控制](https://www.jianshu.com/p/53738efa2000)
