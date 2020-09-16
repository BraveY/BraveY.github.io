$(function () {
  const isSnackbar = GLOBAL_CONFIG.Snackbar !== undefined
  const $pageHead = $('#page-header')
  const $rightside = $('#rightside')
  const $body = $('body')

  /**
   * 當menu過多時，自動適配，避免UI錯亂
    * 傳入 1 sidebar打開時
   * 傳入 2 正常狀態下
   */
  var blogNameWidth = $('#blog_name').width()
  var menusWidth = $('.menus').width()
  var sidebarWidth = $('#sidebar').width()

  function isAdjust (n) {
    var t
    if (n === 1) {
      t = blogNameWidth + menusWidth > $pageHead.width() - sidebarWidth - 20
    } else if (n === 2) {
      t = blogNameWidth + menusWidth > $pageHead.width() - 20
    }

    if (t) headerAdjust()
    else headerAdjustBack()
  }

  function headerAdjust () {
    $pageHead.find('.toggle-menu').addClass('is-visible-inline')
    $pageHead.find('.menus_items').addClass('is_invisible')
    $pageHead.find('#search_button span').addClass('is_invisible')
  }

  function headerAdjustBack () {
    $pageHead.find('.toggle-menu').removeClass('is-visible-inline')
    $pageHead.find('.menus_items').removeClass('is_invisible')
    $pageHead.find('#search_button span').removeClass('is_invisible')
  }

  // 初始化header
  function initAjust () {
    if (window.innerWidth < 768) headerAdjust()
    else isAdjust(2)
  }

  initAjust()
  $('#page-header').css({ opacity: '1', animation: 'headerNoOpacity 1s' })

  $(window).on('resize', function () {
    if ($('#sidebar').hasClass('tocOpenPc') && $pageHead.hasClass('fixed')) {
      isAdjust(1)
    } else {
      initAjust()
    }
  })

  /**
   * windows時 設置主頁top_img 為 fixed
   */
  if (GLOBAL_CONFIG_SITE.isHome) {
    var isSafari = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent)
    if (!isSafari && isDesktop()) {
      $('#nav').css('background-attachment', 'fixed')
    }
  }

  /**
   * 進入post頁sidebar處理
   */
  if (GLOBAL_CONFIG_SITE.isPost) {
    if (window.innerWidth > 1024 && $('#toggle-sidebar').hasClass('on')) {
      setTimeout(function () {
        openSidebar()
      }, 400)
    }
  }

  /**
   * 點擊左下角箭頭,顯示sidebar
   */

  function closeSidebar () {
    $('#sidebar').removeClass('tocOpenPc')
    $('.menus').animate({
      paddingRight: 0
    }, 400)
    $('#body-wrap').animate({
      paddingLeft: 0
    }, 400)
    $('#sidebar').animate({
      left: '-300px'
    }, 400)
    $('#toggle-sidebar').css({
      transform: 'rotateZ(0deg)',
      color: '#1F2D3D',
      opacity: '1'
    })
    setTimeout(function () {
      isAdjust(2)
    }, 400)
  }

  function openSidebar () {
    $('#sidebar').addClass('tocOpenPc')
    $('.menus').animate({
      paddingRight: 300
    }, 400)
    $('#body-wrap').animate({
      paddingLeft: 300
    }, 400)
    $('#sidebar').animate({
      left: 0
    }, 400)
    $('#toggle-sidebar').css({
      transform: 'rotateZ(180deg)',
      color: '#99a9bf',
      opacity: '1'
    })
    var isAdjustTimeCount = window.setInterval(function () {
      if ($pageHead.hasClass('fixed')) isAdjust(1)
      else isAdjust(2)
    }, 100)
    setTimeout(function () {
      clearInterval(isAdjustTimeCount)
    }, 400)
  }

  $('#toggle-sidebar').on('click', function () {
    var isOpen = $(this).hasClass('on')
    isOpen ? $(this).removeClass('on') : $(this).addClass('on')
    if (isOpen) {
      closeSidebar()
    } else {
      openSidebar()
    }
  })

  /**
   * 手機menu和toc按鈕點擊
   * 顯示menu和toc的sidebar
   */

  var $toggleMenu = $('.toggle-menu')
  var $mobileSidevarMenus = $('#mobile-sidebar-menus')
  var $mobileTocButton = $('#mobile-toc-button')
  var $menuMask = $('#menu_mask')

  function openMobileSidebar (name) {
    sidebarPaddingR()
    $('body').css('overflow', 'hidden')
    $menuMask.fadeIn()

    if (name === 'menu') {
      $toggleMenu.removeClass('close').addClass('open')
      $mobileSidevarMenus.css('transform', 'translate3d(-100%,0,0)')
      var $mobileSidevarMenusChild = $mobileSidevarMenus.children()
      for (let i = 0; i <= $mobileSidevarMenusChild.length; i++) {
        const duration = i / 5 + 0.2
        $mobileSidevarMenusChild.eq(i).css('animation', 'sidebarItem ' + duration + 's')
      }
    }

    if (name === 'toc') {
      $mobileTocButton.removeClass('close').addClass('open')
      $('#sidebar').addClass('tocOpenMobile')
      $('#sidebar').css({ transform: 'translate3d(-100%,0,0)', left: '' })
    }
  }

  function closeMobileSidebar (name) {
    $('body').css({ overflow: '', 'padding-right': '' })
    $menuMask.fadeOut()

    if (name === 'menu') {
      $toggleMenu.removeClass('open').addClass('close')
      $mobileSidevarMenus.css('transform', '')
      $('#mobile-sidebar-menus > div,#mobile-sidebar-menus > hr').css('animation', '')
    }

    if (name === 'toc') {
      $mobileTocButton.removeClass('open').addClass('close')
      $('#sidebar').removeClass('tocOpenMobile')
      $('#sidebar').css({ transform: '' })
    }
  }

  $toggleMenu.on('click', function () {
    openMobileSidebar('menu')
  })

  $mobileTocButton.on('click', function () {
    openMobileSidebar('toc')
  })

  $menuMask.on('click touchstart', function (e) {
    if ($toggleMenu.hasClass('open')) {
      closeMobileSidebar('menu')
    }
    if ($mobileTocButton.hasClass('open')) {
      closeMobileSidebar('toc')
    }
  })

  $(window).on('resize', function (e) {
    if (!$toggleMenu.is(':visible')) {
      if ($toggleMenu.hasClass('open')) closeMobileSidebar('menu')
    }
  })

  const mql = window.matchMedia('(max-width: 1024px)')
  mql.addListener(function (ev) {
    if (ev.matches) {
      if ($('#sidebar').hasClass('tocOpenPc')) closeSidebar()
    } else {
      if ($('#toggle-sidebar').hasClass('on')) openSidebar()
      if ($mobileTocButton.hasClass('open')) closeMobileSidebar('toc')
    }
  })

  /**
   * 首頁top_img底下的箭頭
   */
  $('#scroll_down').on('click', function () {
    scrollTo('#content-inner')
  })

  /**
   * BOOKMARK 書簽
   */
  $('#bookmark-it').on('click', function () {
    if (window.sidebar && window.sidebar.addPanel) { // Mozilla Firefox Bookmark
      window.sidebar.addPanel(document.title, window.location.href, '')
    } else if (window.external && ('AddFavorite' in window.external)) { // IE Favorite
      window.external.AddFavorite(location.href, document.title)
    } else if (window.opera && window.print) { // Opera Hotlist
      this.title = document.title
      return true
    } else { // webkit - safari/chrome
      if (isSnackbar) {
        var bookmarkText = GLOBAL_CONFIG.Snackbar.bookmark.message_prev + ' ' + (navigator.userAgent.toLowerCase().indexOf('mac') !== -1 ? 'Command/Cmd' : 'CTRL') + '+ D ' + GLOBAL_CONFIG.Snackbar.bookmark.message_next + '.'
        snackbarShow(bookmarkText)
      } else {
        alert(GLOBAL_CONFIG.bookmark.message_prev + ' ' + (navigator.userAgent.toLowerCase().indexOf('mac') !== -1 ? 'Command/Cmd' : 'CTRL') + '+ D ' + GLOBAL_CONFIG.bookmark.message_next + '.')
      }
    }
  })

  /**
   * 代碼
   * 只適用於Hexo默認的代碼渲染
   */
  const $figureHighlight = $('figure.highlight')

  if ($figureHighlight.length) {
    const isHighlightCopy = GLOBAL_CONFIG.highlightCopy
    const isHighlightLang = GLOBAL_CONFIG.highlightLang
    const isHighlightShrink = GLOBAL_CONFIG.highlightShrink

    if (isHighlightCopy || isHighlightLang || isHighlightShrink !== 'none') {
      $figureHighlight.wrap('<div class="code-area-wrap"></div>').before('<div class="highlight-tools"></div>')
    }

    /**
   * 代碼收縮
   */
    const $highlightTools = $('.highlight-tools')
    if (isHighlightShrink === 'true') {
      $highlightTools.append('<i class="fa fa-angle-down code-expand code-closed" aria-hidden="true"></i>')
    } else if (isHighlightShrink === 'false') {
      $highlightTools.append('<i class="fa fa-angle-down code-expand" aria-hidden="true"></i>')
    }

    $(document).on('click', '.highlight-tools >.code-expand', function () {
      var $table = $(this).parent().next()
      if ($(this).hasClass('code-closed')) {
        $table.css('display', 'block')
        $(this).removeClass('code-closed')
      } else {
        $table.css('display', 'none')
        $(this).addClass('code-closed')
      }
    })

    /**
    * 代碼語言
  */
    if (isHighlightLang) {
      var langNameIndex, langName
      $figureHighlight.each(function () {
        langNameIndex = langName = $(this).attr('class').split(' ')[1]
        if (langNameIndex === 'plain') langName = 'Code'
        $(this).prev().append('<div class="code-lang">' + langName + '</div>')
      })
    }
    /**
   * 代碼copy
   * copy function
   */
    if (isHighlightCopy) {
      $highlightTools.append('<div class="copy-notice"></div><i class="fa fa-clipboard" aria-hidden="true"></i>')
      var copy = function (text, ctx) {
        if (document.queryCommandSupported && document.queryCommandSupported('copy')) {
          try {
            document.execCommand('copy') // Security exception may be thrown by some browsers.
            if (isSnackbar) {
              snackbarShow(GLOBAL_CONFIG.copy.success)
            } else {
              $(ctx).prev('.copy-notice')
                .text(GLOBAL_CONFIG.copy.success)
                .animate({
                  opacity: 1,
                  right: 30
                }, 450, function () {
                  setTimeout(function () {
                    $(ctx).prev('.copy-notice').animate({
                      opacity: 0,
                      right: 0
                    }, 650)
                  }, 400)
                })
            }
          } catch (ex) {
            if (isSnackbar) {
              snackbarShow(GLOBAL_CONFIG.copy.success)
            } else {
              $(ctx).prev('.copy-notice')
                .text(GLOBAL_CONFIG.copy.error)
                .animate({
                  opacity: 1,
                  right: 30
                }, 650, function () {
                  setTimeout(function () {
                    $(ctx).prev('.copy-notice').animate({
                      opacity: 0,
                      right: 0
                    }, 650)
                  }, 400)
                })
              return false
            }
          }
        } else {
          if (isSnackbar) {
            snackbarShow(GLOBAL_CONFIG.copy.noSupport)
          } else {
            $(ctx).prev('.copy-notice').text(GLOBAL_CONFIG.copy.noSupport)
          }
        }
      }

      // click events
      $(document).on('click', '.highlight-tools>.fa-clipboard', function () {
        var selection = window.getSelection()
        var range = document.createRange()
        range.selectNodeContents($(this).parent().next().find('.code pre')[0])
        selection.removeAllRanges()
        selection.addRange(range)
        var text = selection.toString()
        copy(text, this)
        selection.removeAllRanges()
      })
    }
  }

  /**
 * PhotoFigcaption
 */
  function addPhotoFigcaption () {
    var images = $('#article-container img')
    images.each(function (i, o) {
      var $this = $(o)
      if ($this.attr('alt')) {
        var t = $('<div class="img-alt is-center">' + $this.attr('alt') + '</div>')
        $this.after(t)
      }
    })
  }
  if (GLOBAL_CONFIG.isPhotoFigcaption) addPhotoFigcaption()

  /**
   * justified-gallery 圖庫排版
   */
  var $justifiedGallery = $('.justified-gallery')
  var isJustifiedGallery = false
  if ($justifiedGallery.length) {
    isJustifiedGallery = true
    var $imgList = $justifiedGallery.find('img')
    $imgList.unwrap()
    if ($imgList.length) {
      $imgList.each(function (i, o) {
        if ($(o).attr('data-src')) $(o).attr('src', $(o).attr('data-src'))
        $(o).wrap('<div></div>')
      })
    }
    $('head').append('<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/justifiedGallery/dist/css/justifiedGallery.min.css">')
    loadScript('https://cdn.jsdelivr.net/npm/justifiedGallery/dist/js/jquery.justifiedGallery.min.js', function () {
      initJustifiedGallery($justifiedGallery)
    })

    var initJustifiedGallery = function (selector) {
      selector.each(function (i, o) {
        if ($(this).is(':visible')) {
          $(this).justifiedGallery({
            rowHeight: 220,
            margins: 4
          })
        }
      })
    }
  }

  /**
   * fancybox和 mediumZoom
   */

  var isMediumZoom = GLOBAL_CONFIG.medium_zoom
  var isFancybox = GLOBAL_CONFIG.fancybox
  if (isFancybox) {
    var images = $('#article-container img:not(.gallery-group-img)').not($('a>img'))
    images.each(function (i, o) {
      var lazyloadSrc = $(o).attr('data-src') ? $(o).attr('data-src') : $(o).attr('src')
      $(o).wrap(
        '<a href="' +
        lazyloadSrc +
        '" data-fancybox="group" data-caption="' +
        $(o).attr('alt') +
        '" class="fancybox"></a>'
      )
    })

    $().fancybox({
      selector: '[data-fancybox]',
      loop: true,
      transitionEffect: 'slide',
      protect: true,
      buttons: ['slideShow', 'fullScreen', 'thumbs', 'close']
    })
  } else if (isMediumZoom) {
    const zoom = mediumZoom(document.querySelectorAll('#article-container :not(a)>img'))
    zoom.on('open', function (event) {
      var photoBg = $(document.documentElement).attr('data-theme') === 'dark' ? '#121212' : '#fff'
      zoom.update({
        background: photoBg
      })
    })
  }

  /**
   * 滾動處理
   */
  var initTop = 0
  $(window).scroll(function (event) {
    var currentTop = $(this).scrollTop()
    var isUp = scrollDirection(currentTop)
    if (currentTop > 56) {
      if (isUp) {
        if ($pageHead.hasClass('visible')) $pageHead.removeClass('visible')
      } else {
        if (!$pageHead.hasClass('visible')) $pageHead.addClass('visible')
      }
      $pageHead.addClass('fixed')
      if ($rightside.css('opacity') === '0') {
        $rightside.css({ opacity: '1', transform: 'translateX(-38px)' })
      }
    } else {
      if (currentTop === 0) {
        $pageHead.removeClass('fixed').removeClass('visible')
      }
      $rightside.css({ opacity: '', transform: '' })
    }
  })

  // find the scroll direction
  function scrollDirection (currentTop) {
    var result = currentTop > initTop // true is down & false is up
    initTop = currentTop
    return result
  }

  /**
   * 點擊滾回頂部
   */
  $('#go-up').on('click', function () {
    scrollTo('body')
  })

  /**
   *  toc
   */

  if (GLOBAL_CONFIG_SITE.isPost && GLOBAL_CONFIG_SITE.isSidebar) {
    $('.toc-child').hide()

    // main of scroll
    $(window).scroll(throttle(function (event) {
      var currentTop = $(this).scrollTop()
      scrollPercent(currentTop)
      findHeadPosition(currentTop)
      autoScrollToc(currentTop)
    }, 100))

    // scroll
    $('.toc-link').on('click', function (e) {
      if (window.innerWidth <= 1024) {
        closeMobileSidebar('toc')
      } else {
        e.preventDefault()
        scrollTo($(this).attr('href'))
      }
    })

    // expand toc-item
    var expandToc = function ($item) {
      if ($item.is(':visible')) {
        return
      }
      $item.fadeIn(400)
    }

    var scrollPercent = function (currentTop) {
      var docHeight = $('#article-container').height()
      var winHeight = $(window).height()
      var contentMath = (docHeight > winHeight) ? (docHeight - winHeight) : ($(document).height() - winHeight)
      var scrollPercent = (currentTop) / (contentMath)
      var scrollPercentRounded = Math.round(scrollPercent * 100)
      var percentage = (scrollPercentRounded > 100) ? 100
        : (scrollPercentRounded <= 0) ? 0
          : scrollPercentRounded
      $('.progress-num').text(percentage)
      $('.sidebar-toc__progress-bar').animate({
        width: percentage + '%'
      }, 100)
    }

    // function updateAnchor (anchor) {
    //   if (window.history.replaceState && anchor !== window.location.hash) {
    //     window.history.replaceState(null, null, anchor)
    //   }
    // }

    // find head position & add active class
    // DOM Hierarchy:
    // ol.toc > (li.toc-item, ...)
    // li.toc-item > (a.toc-link, ol.toc-child > (li.toc-item, ...))
    var findHeadPosition = function (top) {
    // assume that we are not in the post page if no TOC link be found,
    // thus no need to update the status
      if ($('.toc-link').length === 0) {
        return false
      }

      var list = $('#article-container').find('h1,h2,h3,h4,h5,h6')
      var currentId = ''
      list.each(function () {
        var head = $(this)
        if (top > head.offset().top - 25) {
          currentId = '#' + $(this).attr('id')
        }
      })

      if (currentId === '') {
        $('.toc-link').removeClass('active')
        $('.toc-child').hide()
      }

      var currentActive = $('.toc-link.active')
      if (currentId && currentActive.attr('href') !== currentId) {
        // updateAnchor(currentId)

        $('.toc-link').removeClass('active')

        var _this = $('.toc-link[href="' + currentId + '"]')
        _this.addClass('active')

        var parents = _this.parents('.toc-child')
        // Returned list is in reverse order of the DOM elements
        // Thus `parents.last()` is the outermost .toc-child container
        // i.e. list of subsections
        var topLink = (parents.length > 0) ? parents.last() : _this
        expandToc(topLink.closest('.toc-item').find('.toc-child'))
        topLink
          // Find all top-level .toc-item containers, i.e. sections
          // excluding the currently active one
          .closest('.toc-item').siblings('.toc-item')
          // Hide their respective list of subsections
          .find('.toc-child').hide()
      }
    }

    var autoScrollToc = function (currentTop) {
      if ($('.toc-link').hasClass('active')) {
        var activePosition = $('.active').offset().top
        var sidebarScrolltop = $('#sidebar .sidebar-toc__content').scrollTop()
        if (activePosition > (currentTop + $(window).height() - 100)) {
          $('#sidebar .sidebar-toc__content').scrollTop(sidebarScrolltop + 100)
        }
        if (activePosition < currentTop + 100) {
          $('#sidebar .sidebar-toc__content').scrollTop(sidebarScrolltop - 100)
        }
      }
    }
  }

  /**
   * 閲讀模式
   */
  $('#readmode').click(function () {
    $('body').toggleClass('read-mode')
  })

  /**
   * 字體調整
   */
  $('#font_plus').click(function () {
    $body.css('font-size', parseFloat($body.css('font-size')) + 1)
  })
  $('#font_minus').click(function () {
    $body.css('font-size', parseFloat($body.css('font-size')) - 1)
  })

  /**
   * menu
   * 側邊欄sub-menu 展開/收縮
   * 解決menus在觸摸屏下，滑動屏幕menus_item_child不消失的問題（手機hover的bug)
   */
  $('#mobile-sidebar-menus .menus-expand').on('click', function () {
    if ($(this).hasClass('menus-closed')) {
      $(this).parents('.menus_item').find('.menus_item_child').slideDown()
      $(this).removeClass('menus-closed')
    } else {
      $(this).parents('.menus_item').find('.menus_item_child').slideUp()
      $(this).addClass('menus-closed')
    }
  })

  $(window).on('touchmove', function (e) {
    var $menusChild = $('#page-header .menus_item_child')
    if ($menusChild.is(':visible')) {
      $menusChild.css('display', 'none')
    }
  })

  /**
   * rightside 點擊設置 按鈕 展開
   */
  $('#rightside_config').on('click', function () {
    if ($('#rightside-config-hide').hasClass('rightside-in')) {
      $('#rightside-config-hide').removeClass('rightside-in').addClass('rightside-out')
    } else {
      $('#rightside-config-hide').removeClass('rightside-out').addClass('rightside-in')
    }
  })

  /**
   * 複製時加上版權信息
   */
  var copyright = GLOBAL_CONFIG.copyright
  if (copyright !== undefined) {
    document.body.oncopy = function (event) {
      event.preventDefault()
      let textFont; const copyFont = window.getSelection(0).toString()
      if (copyFont.length > 45) {
        textFont = copyFont + '\n' + '\n' + '\n' +
          copyright.languages.author + '\n' +
          copyright.languages.link + window.location.href + '\n' +
          copyright.languages.source + '\n' +
          copyright.languages.info
      } else {
        textFont = copyFont
      }
      if (event.clipboardData) {
        return event.clipboardData.setData('text', textFont)
      } else {
        // 兼容IE
        return window.clipboardData.setData('text', textFont)
      }
    }
  }

  /**
   * Darkmode
   */
  var isFontAwesomeV5 = GLOBAL_CONFIG.isFontAwesomeV5
  var $darkModeButtom = $('#darkmode')
  if (typeof autoChangeMode !== 'undefined') {
    document.documentElement.getAttribute('data-theme') === 'dark' ? changeLightIcon() : changeDarkIcon()
  }

  function changeLightIcon () {
    isFontAwesomeV5 ? $darkModeButtom.removeClass('fa-moon').addClass('fa-sun') : $darkModeButtom.removeClass('fa-moon-o').addClass('fa-sun-o')
  }

  function changeDarkIcon () {
    isFontAwesomeV5 ? $darkModeButtom.removeClass('fa-sun').addClass('fa-moon') : $darkModeButtom.removeClass('fa-sun-o').addClass('fa-moon-o')
  }

  function switchReadMode () {
    var nowMode = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'
    if (nowMode === 'light') {
      changeLightIcon()
      activateDarkMode()
      Cookies.set('theme', 'dark', { expires: 2 })
      if (isSnackbar) snackbarShow(GLOBAL_CONFIG.Snackbar.day_to_night)
    } else {
      changeDarkIcon()
      activateLightMode()
      Cookies.set('theme', 'light', { expires: 2 })
      if (isSnackbar) snackbarShow(GLOBAL_CONFIG.Snackbar.night_to_day)
    }
  }

  $darkModeButtom.click(function () {
    switchReadMode()
    try {
      utterancesTheme()
    } catch (e) {
      return false
    }
  })

  /**
   * 網頁運行時間
   */
  if (GLOBAL_CONFIG.runtime) {
    // get user config
    var $runtimeCount = $('#webinfo-runtime-count')
    var startDate = $runtimeCount.attr('start_date')
    var showDateTime = function () {
      var BirthDay = new Date(startDate)
      var today = new Date()
      var timeold = (today.getTime() - BirthDay.getTime())
      var daysold = Math.floor(timeold / (24 * 60 * 60 * 1000))
      $runtimeCount.text(daysold + ' ' + GLOBAL_CONFIG.runtime_unit)
    }
    var interval
    showDateTime()
    clearInterval(interval)
    interval = setInterval(showDateTime, 10000)
  }

  /**
   * 百度推送
   */
  if (GLOBAL_CONFIG.baiduPush) {
    (function () {
      var bp = document.createElement('script')
      var curProtocol = window.location.protocol.split(':')[0]
      if (curProtocol === 'https') {
        bp.src = 'https://zz.bdstatic.com/linksubmit/push.js'
      } else {
        bp.src = 'http://push.zhanzhang.baidu.com/push.js'
      }
      var s = document.getElementsByTagName('script')[0]
      s.parentNode.insertBefore(bp, s)
    })()
  }

  /**
   * tag-hide
   */
  var $hideInline = $('.hide-button')
  if ($hideInline.length) {
    $hideInline.on('click', function (e) {
      e.preventDefault()
      var $this = $(this)
      var $hideContent = $(this).next('.hide-content')
      $this.toggleClass('open')
      $hideContent.toggle()
      if ($this.hasClass('open')) {
        if (isJustifiedGallery && $hideContent.find('.justified-gallery').length > 0) {
          initJustifiedGallery($hideContent.find('.justified-gallery'))
        }
      }
    })
  }
})
